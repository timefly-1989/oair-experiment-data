# OAIR experiment data

This repository accompanies **Image-Anchored Refinement of Open Garment Meshes: An Initializer-Relative Framework from Synthetic to Real Observations** by Shuai Liu, Hui Li, and Jin Qian. The files correspond to the 2026-11-24 manuscript.

## Contents

| File | Contents |
| --- | --- |
| [table_index.csv](table_index.csv) | Index of the 14 manuscript tables, with captions and LaTeX labels. Each linked filename has one CSV with the values as printed. Where reported, seed SD is in a separate column. |
| [syn600_case_metrics.csv](syn600_case_metrics.csv) | 6,324 model, recipe, seed and condition rows from 68 garments across 67 synthetic body groups. Distances are in meters. Seed 0 identifies shared initializer rows. |
| [syn600_paired_contrasts.csv](syn600_paired_contrasts.csv) | DDL and Transolver++ minus OAIR composite error for the validation-selected recipe, with paired seed differences and Bonferroni-adjusted 97.5% intervals, in mm. |
| [syn30_component_metrics.csv](syn30_component_metrics.csv) | 3,667 case-level rows across 193 garments and 188 synthetic body groups for the initializer and six scale-augmented component variants. Distances are in mm. |
| [captured_coverage_aggregates.csv](captured_coverage_aggregates.csv) | Full-precision archived coverage aggregates for the pilot and CAP-20. There are 675 eligible supplied-mask patches and 670 valid predicted-input patches across 20 distinct holdout subjects. Distances are nominal mm. |
| [cap20_supplied_seed_metrics.csv](cap20_supplied_seed_metrics.csv) | 18 equal-subject seed aggregates for clean CAP-20 supplied-mask evaluations. Each seed averages within subjects, then across the 20 subjects. |
| [cap20_mesh_diagnostics.csv](cap20_mesh_diagnostics.csv) | Orientation reversal and collapse as fractions, with the per-mesh log-stretch p95 summary for both CAP-20 input tracks. Table 11 reports reversal and collapse as percentages. |
| [initializer_references_mm.json](initializer_references_mm.json) | Clean initializer values used for the three-domain summary figure. |

SYN-600 averages cases within a body, then equally across the 67 body groups and five training seeds. The synthetic composite error is surface CD + 0.35 × boundary CD. SYN-30 Tables 7 and 8 use case-weighted means. The numeric portions of Tables 9, 12 and 13 are manuscript exports at their reported precision. The SYN-30 per-case file supports the mean and worst opening columns and a subset of the Table 8 composite results.

## Verify the results

From the repository directory, use Python 3.13 with NumPy.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The script verifies file hashes and table record counts, regenerates Table 6 and the paired SYN-600 bootstrap statistics using 6,000 crossed body/seed draws with RNG seed 1110, verifies the supported SYN-30 component summaries, and checks the 4D-DRESS table entries against the released aggregates. This is a CPU statistical check from saved measurements. Full inference requires the original datasets, pretrained assets and refiner weights.

## Dataset access and citation

Use the official distribution pages for [GarmentCodeData](https://igl.ethz.ch/projects/GarmentCodeData/), [CloSe](https://virtualhumans.mpi-inf.mpg.de/close3dv24/), and [4D-DRESS](https://4d-dress.ait.ethz.ch/). The respective provider terms govern access and use. This repository contains derived numerical results. Its synthetic case and group labels are local to each CSV. The real-data files contain cohort and seed aggregates. Refiner weights are available from the corresponding author on reasonable request as stated in the manuscript.

Please cite the data snapshot used for your work. Author and repository metadata are in [CITATION.cff](CITATION.cff).
