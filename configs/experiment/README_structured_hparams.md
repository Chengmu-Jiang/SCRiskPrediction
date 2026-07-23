# Structured Hyperparameter Experiments

Run any file in this folder with:

```powershell
python .\src\train.py experiment=<config_name_without_yaml>
```

The training defaults are in `configs/train.yaml`, the base model defaults are in
`configs/model/structured_risk.yaml`, and these experiment files override selected
global parameters.

## Repeated TimeSeriesSplit CV

Set `repeat_seeds` to run the whole TimeSeriesSplit CV multiple times in one command.
Each seed runs a full CV under `repeat_XX/fold_YY`, then the parent output directory
stores both repeat-level and all-fold mean/std summaries.

```yaml
repeat_seeds: "42,2026,3407,8703,3818"
```

Generated parent-level files:

| File | Contents |
| --- | --- |
| `repeated_time_series_cv_summary.csv` | One row per repeat/split/metric; `value` is that repeat's fold-mean metric. |
| `repeated_time_series_cv_aggregate_metrics.csv` | Mean and std across repeat-level values. |
| `repeated_time_series_cv_fold_summary.csv` | All fold rows across all repeats. |
| `repeated_time_series_cv_25fold_aggregate_metrics.csv` | Mean and std computed directly over all folds across repeats. |
| `repeated_time_series_cv_summary.json` | JSON payload with seeds, repeat outputs, and aggregate metrics. |

## Recommended baseline

| Config | Purpose |
| --- | --- |
| `structured_hp_baseline` | Current tuned baseline: alignment, graph smoothness, deviation diffusion, compact 64-dim model. |
| `structured_pipeline_8703` | Backward-compatible historical 8703 setting with `align_weight=0.05`. |

## Propagation model baselines

These configs keep the same supervised pipeline and use `smooth_weight=0.0` for all
models. Outputs are written under `outputs/structured_baselines/`.

| Config | propagation_model | Output subdirectory |
| --- | --- | --- |
| `structured_baseline_reaction_diffusion` | `reaction_diffusion` | `reaction_diffusion` |
| `structured_baseline_gcn` | `gcn` | `gcn` |
| `structured_baseline_gat` | `gat` | `gat` |
| `structured_baseline_dcrnn_single` | `dcrnn` | `dcrnn_single` |
| `structured_baseline_dcrnn_bi` | `dcrnn` with bidirectional diffusion | `dcrnn_bi` |
| `structured_baseline_agcrn` | `agcrn` with adaptive graph generation | `agcrn` |
| `structured_baseline_lstm` | `lstm` | `lstm` |
| `structured_baseline_gru_gated_reaction_diffusion` | `gru_gated_reaction_diffusion` | `gru_gated_reaction_diffusion` |
| `structured_baseline_lstm_gated_reaction_diffusion` | `lstm_gated_reaction_diffusion` | `lstm_gated_reaction_diffusion` |
| `structured_baseline_gru_z_gated_reaction_diffusion` | `gru_z_gated_reaction_diffusion` | `gru_z_gated_reaction_diffusion` |
| `structured_baseline_lstm_z_gated_reaction_diffusion` | `lstm_z_gated_reaction_diffusion` | `lstm_z_gated_reaction_diffusion` |
| `structured_baseline_lstm_component_gated_reaction_diffusion` | `lstm_component_gated_reaction_diffusion` with learnable component scale | `lstm_component_gated_reaction_diffusion` |
| `structured_baseline_lstm_component_gated_reaction_diffusion_no_scale` | `lstm_component_gated_reaction_diffusion` with `component_scale_mode=none` | `lstm_component_gated_reaction_diffusion_no_scale` |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion` | `lstm_component_fusion_gated_reaction_diffusion` with learnable component scale and RD fusion gate | `lstm_component_fusion_gated_reaction_diffusion` |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_no_scale` | `lstm_component_fusion_gated_reaction_diffusion` with `component_scale_mode=none` and RD fusion gate | `lstm_component_fusion_gated_reaction_diffusion_no_scale` |

## Fusion-gated reaction-diffusion stability experiments

These configs target the instability observed in the fusion-gated LSTM reaction-diffusion
variant. They use `rd_fusion_gate_init=0.95` and `early_stopping_patience=20` unless
otherwise noted, so the RD component is initially preserved and validation noise is less
likely to stop training too early.

| Config | Main change | Purpose |
| --- | --- | --- |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_000` | `align_weight=0.0` | Check whether LLM alignment hurts supervised RMSE. |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_010` | `align_weight=0.1` | Low-alignment candidate for the main improved model. |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_030` | `align_weight=0.3` | Mid-alignment sensitivity point. |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_010_low_dropout` | `dropout=graph_dropout=eval_dropout=align_dropout=0.1` | Test whether 0.3 dropout is underfitting small folds. |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_010_one_hop` | `enable_two_hop_diffusion=false` | Remove the noisier two-hop diffusion term. |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_010_csi_050` | `component_scale_init=0.5` | Give RD components more initial magnitude. |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_stable_v1` | low dropout, one-hop diffusion, `component_scale_init=0.5`, `align_weight=0.1` | Combined stability candidate after the single-factor checks. |

## Alignment loss weight

Use these to analyze the contribution and sensitivity of LLM-structured contrastive alignment.

| Config | align_weight |
| --- | ---: |
| `structured_hp_align_000` | 0.00 |
| `structured_hp_align_005` | 0.05 |
| `structured_hp_align_010` | 0.10 |
| `structured_hp_align_030` | 0.30 |
| `structured_hp_align_060` | 0.60 |

## Graph smoothness weight

Use these to analyze graph regularization on industry-chain predictions.

| Config | smooth_weight |
| --- | ---: |
| `structured_hp_smooth_000` | 0.00 |
| `structured_hp_smooth_010` | 0.01 |
| `structured_hp_smooth_050` | 0.05 |
| `structured_hp_smooth_100` | 0.10 |

## Diffusion mode

| Config | diffusion_mode |
| --- | --- |
| `structured_hp_baseline` | `deviation` |
| `structured_hp_diffusion_direct` | `direct` |

## Graph direction

Use these to test whether directed or bidirectional industry-chain propagation better matches the task.

| Config | graph_direction | Purpose |
| --- | --- | --- |
| `structured_hp_graphdir_downstream_to_upstream` | `downstream-to-upstream` | Current default direction. |
| `structured_hp_graphdir_upstream_to_downstream` | `upstream-to-downstream` | Reverse directed propagation. |
| `structured_hp_graphdir_bidirectional` | `bidirectional` | Treats each product-chain edge as two propagation directions. |

## Graph propagation ablations

Use these for fine-grained ablations of the graph propagation update.

| Config | enable_graph_diffusion | enable_two_hop_diffusion | enable_recovery | Purpose |
| --- | --- | --- | --- | --- |
| `structured_hp_baseline` | true | true | true | Full propagation update. |
| `structured_hp_ablate_no_diffusion` | false | true | true | Removes the whole graph diffusion term `h_diff`. |
| `structured_hp_ablate_one_hop_only` | true | false | true | Keeps one-hop diffusion and removes `gamma * two_hop`. |
| `structured_hp_ablate_no_recovery` | true | true | false | Removes the recovery/reaction term `h_react`. |

## Initial h0 ablations

Use these to compare the IGPC prior against non-IGPC node baselines.

| Config | initial_state_mode | Purpose |
| --- | --- | --- |
| `structured_hp_h0_igpc` | `igpc` | Current IGPC-based initialization, used as an explicit baseline. |
| `structured_hp_h0_learnable` | `learnable` | Learns node-specific `h0 = sigmoid(raw_h0)` without IGPC information. |
| `structured_hp_h0_fixed_zero` | `fixed_zero` | Uses a fixed, non-learnable all-zero `h0`. |

## Model capacity

| Config | projection_dim | representation_dim | graph_state_dim |
| --- | ---: | ---: | ---: |
| `structured_hp_baseline` | 64 | 64 | 64 |
| `structured_hp_capacity_128` | 128 | 128 | 128 |
| `structured_hp_capacity_graph256` | 128 | 128 | 256 |

## Contrastive temperature

| Config | align_temperature |
| --- | ---: |
| `structured_hp_temp_003` | 0.03 |
| `structured_hp_baseline` | 0.07 |
| `structured_hp_temp_020` | 0.20 |

## Dropout regularization

| Config | dropout values |
| --- | --- |
| `structured_hp_dropout_010` | `dropout=graph_dropout=eval_dropout=align_dropout=0.1` |
| `structured_hp_baseline` | `dropout=graph_dropout=eval_dropout=align_dropout=0.3` |
| `structured_hp_dropout_050` | `dropout=graph_dropout=eval_dropout=align_dropout=0.5` |

## Example commands

```powershell
python .\src\train.py experiment=structured_hp_baseline
python .\src\train.py experiment=structured_hp_align_000
python .\src\train.py experiment=structured_hp_smooth_000
python .\src\train.py experiment=structured_hp_diffusion_direct
python .\src\train.py experiment=structured_hp_graphdir_downstream_to_upstream
python .\src\train.py experiment=structured_hp_graphdir_upstream_to_downstream
python .\src\train.py experiment=structured_hp_graphdir_bidirectional
python .\src\train.py experiment=structured_hp_ablate_no_diffusion
python .\src\train.py experiment=structured_hp_ablate_one_hop_only
python .\src\train.py experiment=structured_hp_ablate_no_recovery
python .\src\train.py experiment=structured_hp_h0_igpc
python .\src\train.py experiment=structured_hp_h0_learnable
python .\src\train.py experiment=structured_hp_h0_fixed_zero
python .\src\train.py experiment=structured_hp_capacity_128
python .\src\train.py experiment=structured_hp_temp_003
python .\src\train.py experiment=structured_hp_dropout_010
python .\src\train.py experiment=structured_baseline_reaction_diffusion
python .\src\train.py experiment=structured_baseline_gcn
python .\src\train.py experiment=structured_baseline_gat
python .\src\train.py experiment=structured_baseline_dcrnn_single
python .\src\train.py experiment=structured_baseline_dcrnn_bi
python .\src\train.py experiment=structured_baseline_lstm
python .\src\train.py experiment=structured_baseline_gru_gated_reaction_diffusion
python .\src\train.py experiment=structured_baseline_lstm_gated_reaction_diffusion
python .\src\train.py experiment=structured_baseline_gru_z_gated_reaction_diffusion
python .\src\train.py experiment=structured_baseline_lstm_z_gated_reaction_diffusion
python .\src\train.py experiment=structured_baseline_lstm_component_gated_reaction_diffusion
python .\src\train.py experiment=structured_baseline_lstm_component_gated_reaction_diffusion_no_scale
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion_no_scale
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_000
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_010
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_030
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_010_low_dropout
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_010_one_hop
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion_rfgi_095_align_010_csi_050
python .\src\train.py experiment=structured_baseline_lstm_component_fusion_gated_reaction_diffusion_stable_v1
```
