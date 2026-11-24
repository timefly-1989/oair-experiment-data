"""Check released split membership, training schedules, and validation-only recipe selection."""
from collections import defaultdict
import csv
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_runs(stage, run_count):
    runs = csv_rows(f"records/{stage}_training_runs.csv")
    assert len(runs) == run_count
    by_run = defaultdict(list)
    for row in csv_rows(f"records/{stage}_validation_history.csv"):
        by_run[row["run"]].append(row)
    assert set(by_run) == {row["run"] for row in runs}
    for row in runs:
        history = by_run[row["run"]]
        epoch_count, interval = int(row["epochs"]), int(row["validate_every"])
        assert [int(item["epoch"]) for item in history] == list(range(1, epoch_count + 1))
        validations = [item for item in history if item["validation_score"] != ""]
        assert [int(item["epoch"]) for item in validations] == list(range(interval, epoch_count + 1, interval))
        assert isclose(min(float(item["validation_score"]) for item in validations),
                       float(row["best_validation_score"]), rel_tol=1e-12)
        checkpoint = row["checkpoint_sha256"]
        assert len(checkpoint) == 64 and all(c in "0123456789abcdef" for c in checkpoint)
    return {row["run"]: row for row in runs}


def check_syn600(protocol):
    runs = check_runs("syn600", 30)
    measurements = {(r["case"], r["body_group"], r["category"])
                    for r in csv_rows("syn600_case_metrics.csv")}
    holdout = csv_rows("records/syn600_test_membership.csv")
    recorded = {(r["case"], r["body_group"], r["category"]) for r in holdout}
    assert len(holdout) == 68 and len({r["body_group"] for r in holdout}) == 67
    assert measurements == recorded
    scores = defaultdict(list)
    lock = protocol["test_lock"]
    for job in protocol["jobs"]:
        run = runs[job["id"]]
        assert (run["train_cases"], run["validation_cases"], run["updates"]) == ("600", "170", "4800")
        assert run["metric_scale_augmentation"] == str(job["recipe"] == "scale_aug")
        scores[job["model"], job["recipe"]].append(float(run["best_validation_score"]))
    for model, reference in lock["validation_recipe_scores"].items():
        calculated = {recipe: mean(scores[model, recipe]) for recipe in reference}
        assert all(isclose(calculated[recipe], reference[recipe], rel_tol=1e-12)
                   for recipe in calculated)
        assert min(calculated, key=calculated.get) == lock["selected_recipes"][model]
    assert lock["test_used_for_selection"] is False
    print("SYN-600: validated 30 run records, both recipes, and all 68 test memberships.")


def main():
    for row in csv_rows("experiments/source_provenance.csv"):
        data = (ROOT / row["public_file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["release_sha256"], row["public_file"]
        compile(data, row["public_file"], "exec")
    protocols = json.loads((ROOT / "experiments/protocols.json").read_text(encoding="utf-8"))
    check_syn600(protocols["SYN-600"])
    cap20 = check_runs("cap20", 18)
    assert set(cap20) == set(protocols["CAP-20"]["initialization_sha256"])
    for row in cap20.values():
        assert (row["train_cases"], row["validation_cases"], row["updates"]) == ("398", "116", "19104")
    print("CAP-20: validated all 18 adaptation runs and their validation histories.")
    print("Public experiment source checksums and Python syntax verified.")


if __name__ == "__main__":
    main()
