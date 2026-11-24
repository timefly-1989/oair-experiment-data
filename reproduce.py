"""Check the paper tables and reproduce the SYN-600 bootstrap from the archived metrics."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (2110, 2111, 2112, 2113, 2114)


def read(filename):
    with (ROOT / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_close(value, expected, label, tolerance=0.00500001):
    if abs(value - float(expected)) > tolerance:
        raise AssertionError(f"{label}: computed {value}, expected {expected}")


def verify_files_and_tables():
    with (ROOT / "SHA256SUMS").open(encoding="utf-8") as stream:
        for line in stream:
            checksum, filename = line.strip().split("  ", 1)
            actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            if checksum != actual:
                raise AssertionError(f"Checksum mismatch: {filename}")
    tables = read("table_index.csv")
    assert [int(row["table"]) for row in tables] == list(range(1, 15))
    for row in tables:
        assert len(read(row["file"])) == int(row["rows"]), row["file"]
    print("Verified file checksums and all 14 table exports.")


def check_syn600():
    rows = read("syn600_case_metrics.csv")
    assert len(rows) == 6324
    assert len({row["case"] for row in rows}) == 68
    groups = sorted({row["body_group"] for row in rows})
    assert len(groups) == 67
    measurements = ("surface_cd_m", "boundary_cd_m", "composite_m")
    by_body = defaultdict(list)
    for row in rows:
        key = (row["model"], row["recipe"], row["condition"], int(row["seed"]),
               row["body_group"])
        by_body[key].append([float(row[column]) for column in measurements])

    matched = {}
    for row in read("table_06_extension.csv"):
        model = row["Model"]
        recipe = "none" if model == "Initializer" else "no_scale"
        seeds = (0,) if model == "Initializer" else SEEDS
        values = np.array([
            [np.mean(by_body[model, recipe, "clean", seed, group], axis=0)
             for group in groups]
            for seed in seeds
        ])
        per_seed_mm = values.mean(axis=1) * 1000
        for value, column in zip(per_seed_mm.mean(axis=0),
                                 ("Surface CD", "Boundary CD", "Composite mean")):
            check_close(value, row[column], f"{model} {column}")
        if model != "Initializer":
            check_close(stdev(per_seed_mm[:, 2]), row["Composite seed SD"], model)
            matched[model] = values[:, :, 2]
        print(f"SYN-600 {model:13} clean composite {per_seed_mm[:, 2].mean():.2f} mm")

    for row in read("syn600_paired_contrasts.csv"):
        model = row["control"]
        delta = (matched[model] - matched["OAIR"]) * 1000
        rng = np.random.default_rng(1110)
        nseed, ngroups = delta.shape
        draws = []
        for _ in range(6000):
            seeds = rng.integers(nseed, size=nseed)
            bodies = rng.integers(ngroups, size=ngroups)
            draws.append(delta[np.ix_(seeds, bodies)].mean())
        low, high = np.quantile(draws, [0.0125, 0.9875])
        check_close(delta.mean(), row["mean_difference_mm"], model, 1e-9)
        check_close(low, row["adjusted_97_5_lower_mm"], model, 1e-9)
        check_close(high, row["adjusted_97_5_upper_mm"], model, 1e-9)
        for seed, value in zip(SEEDS, delta.mean(axis=1)):
            check_close(value, row[f"seed_{seed}_difference_mm"], model, 1e-9)
        print(f"SYN-600 {model} minus OAIR: {delta.mean():.2f} mm "
              f"(97.5% interval {low:.2f} to {high:.2f} mm)")


def check_syn30():
    rows = read("syn30_component_metrics.csv")
    assert len(rows) == 3667
    assert len({row["case"] for row in rows}) == 193
    assert len({row["body_group"] for row in rows}) == 188
    metrics = ("composite_mm", "mean_opening_mm", "worst_opening_mm")
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["model"], row["seed"]].append(row)
    by_model = defaultdict(list)
    for (model, seed), cases in by_seed.items():
        assert len(cases) == 193, (model, seed)
        by_model[model].append([mean(float(row[metric]) for row in cases)
                                for metric in metrics])

    for row in read("table_07_opening_integrated.csv"):
        _, means, worsts = zip(*by_model[row["Variant"]])
        check_close(mean(means), row["Mean opening"], row["Variant"])
        check_close(mean(worsts), row["Worst opening"], row["Variant"])
    for row in read("table_08_synthetic.csv"):
        model = row["Model / configuration"]
        if model in by_model:
            composite, _, _ = zip(*by_model[model])
            check_close(mean(composite), row["Composite mean"], model)
            if len(composite) > 1:
                check_close(stdev(composite), row["Composite seed SD"], model)
    print("SYN-30: verified mean/worst opening errors and seven composite rows.")


def check_captured():
    full = {(r["track"], r["model"], r["condition"]): r
            for r in read("captured_coverage_aggregates.csv")}
    tracks = {
        "pilot_supplied": "Pilot, supplied mask",
        "cap20_supplied": "CAP-20, supplied mask",
        "cap20_predicted": "CAP-20, predicted input",
    }
    for row in read("table_10_captured.csv"):
        for track, column in tracks.items():
            archived = full[track, row["Model"], "clean"]
            check_close(float(archived["mean_coverage_mm"]), row[column + " mean"], track)
            if row[column + " seed SD"]:
                check_close(float(archived["seed_sd_mm"]), row[column + " seed SD"], track)
    by_model = defaultdict(list)
    for row in read("cap20_supplied_seed_metrics.csv"):
        assert (row["subjects"], row["patches"]) == ("20", "675")
        by_model[row["model"]].append(float(row["coverage_mm"]))
    for model, values in by_model.items():
        assert len(values) == 3
        archived = full["cap20_supplied", model, "clean"]
        check_close(mean(values), archived["mean_coverage_mm"], model, 1e-8)
        check_close(stdev(values), archived["seed_sd_mm"], model, 1e-8)
    for row in read("table_14_robustness.csv"):
        for model in ("OAIR", "OAIR-DN", "DDL", "Transolver++"):
            archived = full["cap20_supplied", model, row["Condition"]]
            check_close(float(archived["mean_coverage_mm"]), row[f"{model} mean"], model)
            check_close(float(archived["seed_sd_mm"]), row[f"{model} seed SD"], model)
    geometry = {(r["model"], r["input"]): r for r in read("cap20_mesh_diagnostics.csv")}
    for row in read("table_11_quality.csv"):
        model, track = row["Model / input"].split("/")
        archived = geometry[model, track]
        check_close(float(archived["orientation_flip_fraction"]) * 100,
                    row["Reversal (%)"], model, 0.00050001)
        check_close(float(archived["collapsed_face_fraction"]) * 100,
                    row["Collapse (%)"], model, 0.00050001)
        check_close(float(archived["edge_log_stretch_p95"]),
                    row["Log stretch p95"], model, 0.00050001)
    print("4D-DRESS: verified reported coverage, seed SDs, corruption results and mesh diagnostics.")


if __name__ == "__main__":
    verify_files_and_tables()
    check_syn600()
    check_syn30()
    check_captured()
