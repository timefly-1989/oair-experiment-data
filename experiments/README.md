# Experiment implementation and run records

This directory contains archived source snapshots for the OAIR study. The root [README](../README.md) describes the independently runnable statistical reproduction. The source files here expose the model, geometric initialization, loss, data loader, and evaluation method behind those measurements.

## Start from the saved results

Run the following at the repository root after installing [requirements.txt](../requirements.txt).

```sh
python reproduce.py
python scripts/check_experiment_records.py
```

The second command validates the public source hashes, the SYN-600 test membership, the choice of augmentation recipe from five validation seeds, and the archived SYN-600 and CAP-20 run schedules and validation histories. With [PyTorch](https://pytorch.org/get-started/locally/) installed, this additional check exercises the original delta-update and ray-projection operators on CPU using synthetic tensors.

```sh
python scripts/check_refiner_components.py
```

## Source and record index

| Stage | Source | Records |
| --- | --- | --- |
| SYN-600 | [Holdout evaluation and analysis](syn600/). Training uses the scan-stage `garment_realism.train` implementation with synthetic-only manifests. | [Training runs](../records/syn600_training_runs.csv), [epoch histories](../records/syn600_validation_history.csv), [test memberships](../records/syn600_test_membership.csv), [seed metrics](../records/syn600_per_seed_metrics.csv), [category metrics](../records/syn600_descriptive_category_metrics.csv). |
| SYN-30 | [Synthetic preparation, model, training and evaluation](syn30/garment_study/). | [Split and budget](../table_03_data.csv), [cases and openings](../syn30_component_metrics.csv), [ablation results](../table_08_synthetic.csv), [additional input tracks](../table_13_synthetic_tracks.csv). |
| CloSe | [Scan preparation and mixed training](close/garment_realism/) with [shared refiner code](close/garment_study/). | [Split and budget](../table_03_data.csv), [reported results](../table_09_scan.csv). |
| CAP-20 | [Mesh reconstruction and metrics](cap20/garment_lab/), [refiner and topology](cap20/garment_study/), [adaptation and coverage evaluation](cap20/garment_realism/). | [Training runs](../records/cap20_training_runs.csv), [epoch histories](../records/cap20_validation_history.csv), [coverage](../captured_coverage_aggregates.csv), [geometry diagnostics](../cap20_mesh_diagnostics.csv). |

[protocols.json](protocols.json) records SYN-600 and CAP-20 seeds, hyperparameters and selection rules. SYN-600 aliases in the test-membership CSV match those in the public per-case file. Provider garment identifiers refer to GarmentCodeData v2, and the reference and input hashes identify the exact source assets. CAP-20 records describe cohort counts, not provider subject identifiers. The checkpoint hashes in the training tables identify the recorded binaries. A validation score is blank for epochs where validation was not scheduled.

## Running the archived experiment entry points

Obtain the provider data from [GarmentCodeData v2](https://igl.ethz.ch/projects/GarmentCodeData/), [CloSe](https://virtualhumans.mpi-inf.mpg.de/close3dv24/), and [4D-DRESS](https://4d-dress.ait.ethz.ch/), and the pretrained frozen body localizer from [NLF](https://github.com/isarandi/nlf/releases) under the respective terms. The experiment used a versioned NLF-L TorchScript export named `nlf_l_multi_0.3.2.patch5.torchscript`. Check compatibility with that export when preparing inputs. The GPU training/evaluation path depends on PyTorch, NumPy, SciPy, Pillow, and trimesh. Depending on the preparation path, torchvision, triangle, pyrender and PyYAML are also needed. The original renders used an EGL setup. The root Python dependency pin is for statistics.

The prepared JSONL manifests reference per-example NPZ caches and contain split, sample, garment and group IDs. [The loader](cap20/garment_realism/dataset.py) defines the required array keys, relative cache-path resolution and SHA-256 verification. Synthetic caches carry complete reference vertices and faces, whereas real caches use observed semantic patch points. Keep test groups out of training and validation.

The example commands show the original argument interfaces once the historical support modules and inputs are available. Run from the repository root. Use absolute manifest and output paths, and choose the device available on your workstation.

```sh
# SYN-600, OAIR, one seed and augmentation disabled
PYTHONPATH=experiments/close:experiments/cap20 python -m garment_realism.train \
  --train /path/to/syn600/train.jsonl --val /path/to/syn600/val.jsonl \
  --out /path/to/runs/OAIR_no_scale_s2110 --architecture evidence_delta \
  --seed 2110 --epochs 8 --lr 0.001 --points 256 --validation-points 1536 \
  --validate-every 2 --device cuda:0

# CAP-20 continuation from a validation-selected captured-pilot checkpoint
PYTHONPATH=experiments/cap20 python -m garment_realism.train \
  --train /path/to/pilot/train.jsonl --val /path/to/pilot/val.jsonl \
  --init /path/to/pilot/OAIR_best.pt --synthetic-replay /path/to/cap20/synthetic_train600.jsonl \
  --out /path/to/runs/A04_raw_s1104 --architecture evidence_delta \
  --seed 1104 --epochs 24 --lr 0.0001 --points 384 --validation-points 1536 \
  --validate-every 4 --scale-augmentation --appearance-augmentation --device cuda:0
```

The local archive contains the main source files above. A full image-to-mesh replay also needs the historical support modules `garment_lab.model` (LocalGraph), `garment_lab.segmentation`, `garment_lab.visualize`, and `garment_expansion.common`, the complete prepared manifests and caches, the frozen RGB parser for predicted-input experiments, and the selected refiner weights. These items were unavailable in the publishable archive, so complete retraining and inference remain unverified. Researchers with provider access can request the missing author-owned experimental artifacts from the corresponding author. Provider assets are distributed by their respective providers.

[source_provenance.csv](source_provenance.csv) gives the original archive and public file hashes. Publication edits replace machine-specific path defaults with explicit paths or environment variables, defer the historical LocalGraph import until full model construction, and normalize line endings. The mathematical operators and recorded measurements are preserved. Original source tree/checkpoint hashes in run records identify the historical runs.
