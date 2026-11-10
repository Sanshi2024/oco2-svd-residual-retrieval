# OCO-2 SVD residual retrieval

Source code and compact results supporting the 10 November 2026 English
manuscript, *Machine-learning retrieval of XCO2 across ocean glint, land
glint, and nadir observation domains*.

The models emulate ACOS Level-2 XCO2. They use 30 masked least-squares
SVD coefficients per band and nine ancillary/prior variables. Global
models use 2015–2019 training, 2020 validation and retrospective
2021–2025 evaluation (2025 is partial).

## Analyses

| Analysis | Source | Compact output |
| --- | --- | --- |
| Validation-selected LightGBM | `lgbm_validation_selection_20261025.py` | `selected_lightgbm_metrics.csv` |
| Independently selected absolute/residual models | `target_specific_search_20261108.py`, `verify_target_specific_20261109.py` | `target_specific_verified_metrics.csv` |
| Prior perturbations | `prior_sensitivity_20261029.py`, `selected_lgbm_controls_20261106.py` | `selected_prior_noise_metrics.csv` |
| Land and latitude diagnostics | `kd_land_diagnostics_20261105.py` | See manuscript |
| East Asian screening and annual evaluation | `run_regional_comparison.py`, `run_sampled_comparison.py` | `east_asia_annual_metrics.csv` |
| Global ND rolling updates | `global_rolling.py` | `global_nd_rolling_verified_metrics.csv` |

For the East Asian annual comparison, the frozen global ND model is
evaluated without regional retraining. The rolling analysis starts from the same global checkpoints and uses
only observations preceding each target quarter. SpT and MLP evaluation
samples were independently drawn.

## Shared preprocessing and model code

Core preprocessing and model code is in `run_three_domain_pipeline.py`,
`run_domain_svd50.py`, `train_domain_svd50.py`, and `search_mlp_secant.py`.
The manuscript uses 30 coefficients per band.

## Reproduction scope

Run `python verify_release.py` to verify checksums and Python syntax
without starting training. The experiment drivers are archival research
code, not an end-to-end installer. Configure the placeholder paths and
provide the matching input files, checkpoints and scalers before use.
No raw granules, sample-level cohorts, weights, fitted bases, scalers,
or manuscript drafts are included. Source products are available from
[NASA GES DISC](https://disc.gsfc.nasa.gov/). No software redistribution
license is granted by this repository notice.
