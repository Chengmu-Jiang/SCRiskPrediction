# Strict rerun experiment protocol

All tuning configurations use one seed and set `evaluate_test: false`. The final
holdout is therefore omitted from metrics and prediction CSV files. Select
hyperparameters using the mean validation RMSE in
`time_series_cv_aggregate_metrics.csv`.

All baseline and ablation configurations inherit `00_base_final.yaml`, use five
distinct seeds, and evaluate the final holdout only in the last expanding-window
fold of each seed.

## 1. Structural screening

Run the structural groups in order: `10_*`, then `11_*`, then `12_*`. After
each group, record that group's lowest-validation-RMSE choice in
`00_base_tuning.yaml`; the next group will inherit it. `01_protocol_current`
is a diagnostic reproduction and is not a tuning candidate.

## 2. PSD screening

Run the PSD groups in order: `20_*`, `21_*`, `22_*`, `23_*`, and `24_*`.
After each group, record that group's lowest-validation-RMSE choice in
`00_base_tuning.yaml`; the next group will inherit it. These reuse the existing
five cached DeepSeek hypotheses; `22_mhyp1`, `22_mhyp3`, and `22_mhyp5` select
cached hypotheses without making API requests.

If the coarse `20_*` search selects zero, run the near-zero `20b_*` refinement.
Only continue to `21_*` through `24_*` when a positive alignment weight wins;
otherwise those parameters are inactive and the searches should be skipped.

`00_base_final.yaml` automatically inherits the frozen tuning values. Do not
change `evaluate_test: true` or the five final seeds.

## 3. Final comparisons and ablations

Run all `30_*` through `34_*` baselines, then `40_*` through `47_*` ablations.
`40_ablation_full` is the full ChainCAST result. `90_final_selected_template`
is an optional explicitly named copy for the final frozen ChainCAST run and
automatically inherits the frozen values.

## Commands

From the repository root:

```powershell
python .\src\train.py experiment=strict_rerun/10_scale010_gate010
```

The three literature/prospective baselines can be run directly with:

```powershell
python .\src\train.py experiment=strict_rerun/34_baseline_grand_t
python .\src\train.py experiment=strict_rerun/34_baseline_gread_t
python .\src\train.py experiment=strict_rerun/34_baseline_prospective_mlp
```

PowerShell batch example:

```powershell
$names = @(
  "10_scale010_gate010",
  "10_scale030_gate030",
  "10_scale050_gate050",
  "10_scale090_gate095"
)
foreach ($name in $names) {
  python .\src\train.py "experiment=strict_rerun/$name"
  if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $name" }
}
```

Never select a configuration using a `test_*` column. Tuning outputs should not
contain such values; final outputs contain test values from one fold per seed.
