# Structured Baseline Results Summary

Source files: `outputs/structured_baselines/<model>/time_series_cv_aggregate_metrics.csv`.

All listed runs use `seed=42`, `smooth_weight=0.0`, `align_weight=0.7`,
`graph_state_dim=64`, `epochs=100`, `lr=3e-4`, and `weight_decay=3e-4`.

## Current Test Metrics

| Model | test MSE | test RMSE | test MAE | test R2 | test Pearson | test Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reaction_diffusion` | 8.613577 +/- 3.012846 | 2.904293 +/- 0.472570 | 2.134005 +/- 0.310595 | 0.276610 +/- 0.253026 | 0.612892 +/- 0.190394 | 0.725874 +/- 0.221768 |
| `gcn` | 10.609272 +/- 3.833795 | 3.210354 +/- 0.615325 | 2.445334 +/- 0.562353 | 0.109007 +/- 0.321972 | 0.359663 +/- 0.559204 | 0.436364 +/- 0.727266 |
| `gat` | 9.199094 +/- 4.243267 | 2.970305 +/- 0.685912 | 2.202034 +/- 0.622612 | 0.227437 +/- 0.356360 | 0.479930 +/- 0.486589 | 0.594406 +/- 0.629798 |
| `dcrnn_single` | 8.199323 +/- 3.358676 | 2.822229 +/- 0.541231 | 1.989297 +/- 0.408818 | 0.311400 +/- 0.282070 | 0.653709 +/- 0.156815 | 0.872727 +/- 0.141891 |
| `dcrnn_bi` | 8.109538 +/- 3.289409 | 2.807736 +/- 0.531688 | 1.934406 +/- 0.409419 | 0.318941 +/- 0.276253 | 0.600998 +/- 0.259252 | 0.711888 +/- 0.309688 |
| `lstm` | 7.724139 +/- 3.586049 | 2.729387 +/- 0.585858 | 2.010671 +/- 0.462273 | 0.351307 +/- 0.301165 | 0.692489 +/- 0.095551 | 0.823776 +/- 0.131145 |
| `gru_gated_reaction_diffusion` | 9.897890 +/- 3.141658 | 3.113758 +/- 0.502994 | 2.303117 +/- 0.408346 | 0.168750 +/- 0.263844 | 0.560244 +/- 0.295770 | 0.660140 +/- 0.364261 |
| `lstm_gated_reaction_diffusion` | 8.196489 +/- 3.328772 | 2.822260 +/- 0.537746 | 2.082794 +/- 0.346454 | 0.311638 +/- 0.279559 | 0.482470 +/- 0.546804 | 0.530070 +/- 0.557764 |

## Pending Gated Variants

| Config | propagation_model | Output subdirectory | Status |
| --- | --- | --- | --- |
| `structured_baseline_gru_z_gated_reaction_diffusion` | `gru_z_gated_reaction_diffusion` | `gru_z_gated_reaction_diffusion` | Pending run |
| `structured_baseline_lstm_z_gated_reaction_diffusion` | `lstm_z_gated_reaction_diffusion` | `lstm_z_gated_reaction_diffusion` | Pending run |
| `structured_baseline_lstm_component_gated_reaction_diffusion` | `lstm_component_gated_reaction_diffusion` with learnable component scale | `lstm_component_gated_reaction_diffusion` | Pending run |
| `structured_baseline_lstm_component_gated_reaction_diffusion_no_scale` | `lstm_component_gated_reaction_diffusion` with `component_scale_mode=none` | `lstm_component_gated_reaction_diffusion_no_scale` | Pending run |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion` | `lstm_component_fusion_gated_reaction_diffusion` with learnable component scale and `rd_fusion_gate_init=0.1` | `lstm_component_fusion_gated_reaction_diffusion` | Pending run |
| `structured_baseline_lstm_component_fusion_gated_reaction_diffusion_no_scale` | `lstm_component_fusion_gated_reaction_diffusion` with `component_scale_mode=none` and `rd_fusion_gate_init=0.1` | `lstm_component_fusion_gated_reaction_diffusion_no_scale` | Pending run |

## Notes

Lower MSE/RMSE/MAE is better. Higher R2/Pearson/Spearman is better. Use this
file as the comparison table for the existing baselines and update the pending
rows after the z-gated, component-gated, and fusion-gated reaction-diffusion runs finish.
