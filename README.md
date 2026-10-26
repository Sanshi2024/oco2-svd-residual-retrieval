# OCO-2 SVD residual retrieval

Research code for prior-referenced XCO2 estimation across ocean glint
(`GL_OCEAN`), land glint (`GL_LAND`) and nadir (`ND`) observations.

## Review status

This is a private, code-only review snapshot aligned with manuscript v15
(26 October 2026). It is not a complete standalone reproduction release.
The manuscript comparator is the original fixed-configuration LightGBM,
not the later validation-selected LightGBM experiment. No new training was
performed to prepare this repository. A redistribution license has not yet
been selected; this private snapshot is not an open-source release.

## Scientific protocol

- Fit domain-specific, uncentered SVD separately in O2-A, weak CO2 and
  strong CO2 using 2015-2019 training spectra.
- Normalize on fixed continuum windows after wavelength alignment and edge
  trimming. Remove flagged channels from both the spectrum and basis rows
  when computing masked least-squares coefficients.
- Use 30 coefficients per band and nine ancillary/prior predictors: 99
  inputs in the exact order recorded in `feature_names.json`.
- Predict reference XCO2 minus its operational prior, then add back that
  same prior. Compare the selected MLP with a fixed LightGBM baseline.
- Global train: 2015-2019; validation: 2020; retrospective tests: 2021-2025.
  The 2025 archive covers only part of the year. Regional ND uses the
  historical 2017-2019 split described in `protocol.json`.

## Files and execution order

Scripts are kept together at repository root so their sibling imports work.
Historical names containing `svd50` do not change the final 30-PC protocol.
Some shared historical modules also support unused Ridge or absolute-target
routes. Those routes are not manuscript results.

| Stage | Entry points |
| --- | --- |
| Sampling and domain assignment | `run_three_domain_pipeline.py` |
| Wavelength processing, SVD and masked LS | `run_domain_svd50.py` |
| 99-feature models and search | `run_prior99_experiment.py`, `search_mlp_secant.py`, `train_prior99_trees.py` |
| Regional ND | `east_asia_prior99.py` |
| Final MLP training audit | `extend_main_epoch500_20261024.py` |
| Million-row component controls | `million_pc_sensitivity_20261024.py` |
| Final ordered sample manifest extraction | `extract_identities.py` |
| Inference from existing 99-feature arrays | `infer_features.py` |
| Figure 5 | `plot_targets.py`, `annual_target_statistics.csv` |

The original orchestration scripts retain author-environment filesystem
paths. Inspect their command-line interfaces and configure the paths before
intentional training. Do not launch an orchestration script just to inspect
the repository. `verify_release.py` performs read-only checks and does not
import the training modules.

## Quick checks

```bash
python verify_release.py
python infer_features.py --root . --verify
python plot_targets.py
```

The last two commands require the scientific dependencies in
`requirements.txt`; plotting writes a local `figures/` directory. The
version list is an observed compatible packaging environment, not an
original-training lockfile. CUDA/PyTorch installation depends on the host.

Inference additionally requires trusted original model/scaler files at the
relative locations in `model_catalog.json`:

```bash
python infer_features.py --root . --cohort GL_OCEAN --family MLP --seed 17   --input your_features.npz --output your_prediction.npz
```

The NPZ input `x` must be an unstandardized, finite N-by-99 float32 array in
catalog order. The last column is the XCO2 prior in ppm. Optional
`sounding_id` and `original_index` arrays are preserved. Model weights and
training scalers are not included, so inference cannot run from this
code-only snapshot alone. Never load untrusted serialized models.

## Included results and omitted artifacts

`fixed_lightgbm_metrics.csv` contains the original fixed-baseline summaries
used by v15; `main_table2.csv` contains the manuscript main comparison.
`annual_target_statistics.csv` contains 33 domain-year aggregates, not raw
soundings. `model_catalog.json` records model hashes and relative artifact
locations without remote-server credentials or addresses.

Raw NASA granules, annual matched HDF5 files, full sounding cohorts, model
weights, fitted SVD binaries, scalers, and manuscript drafts are excluded.
Source data remain available through [NASA GES DISC](https://disc.gsfc.nasa.gov/)
under NASA's access conditions. A complete archival release still needs the
artifact inventory, distribution-license decision and persistent identifier.

Results emulate ACOS Level-2 references rather than independent atmospheric
truth. Later-year evaluation is retrospective, and the regional historical
holdout shares the global spectral representation. Higher-order noise is an
interpretation to test, not a demonstrated causal explanation.

## Provenance

`source_manifest.json` records source hashes and packaging-only edits.
`SHA256.json` verifies included files. None of the checks reselect samples,
refit SVD or retrain regressors. This repository should remain private until
the authors have checked its contents and approved publication.
