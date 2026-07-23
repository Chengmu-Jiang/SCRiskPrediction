from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("numpy")
pytest.importorskip("pandas")

from src.scnews_sentiment.config import structured_namespace_from_cfg

compose = hydra.compose
initialize = hydra.initialize


def test_best_experiment_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_pipeline_8703"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.projection_dim == 64
    assert args.representation_dim == 64
    assert args.graph_state_dim == 64
    assert args.align_weight == 0.05
    assert args.early_stopping_patience == 10
    assert args.news_csv.name == "all_article_scores.csv"
    assert args.nodes_csv.name == "Filtered_Node_List_by_IGPC.csv"
    assert isinstance(args.output_dir, Path)


def test_dry_run_override_is_available() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "dry_run=true",
                "max_rows=10",
                "max_label_rows=10",
                "output_dir=outputs/test_dry_run",
            ],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.dry_run is True
    assert args.max_rows == 10
    assert args.max_label_rows == 10
    assert args.output_dir == Path("outputs/test_dry_run")


def test_repeat_seeds_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_ours"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.repeat_seeds == "42,2026,3407,8703,3818"


def test_dcrnn_baseline_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_dcrnn_bi"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "dcrnn"
    assert args.dcrnn_diffusion_steps == 2
    assert args.dcrnn_bidirectional is True
    assert args.smooth_weight == 0.0
    assert args.output_dir == Path("outputs/structured_baselines/dcrnn_bi")


def test_gat_baseline_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_gat"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "gat"
    assert args.gat_heads == 4
    assert args.graph_state_dim == 64
    assert args.smooth_weight == 0.0


def test_gru_gated_reaction_diffusion_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_gru_gated_reaction_diffusion"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "gru_gated_reaction_diffusion"
    assert args.smooth_weight == 0.0
    assert args.align_weight == 0.7
    assert args.graph_state_dim == 64
    assert args.diffusion_mode == "deviation"
    assert args.output_dir == Path(
        "outputs/structured_baselines/gru_gated_reaction_diffusion"
    )


def test_lstm_gated_reaction_diffusion_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_lstm_gated_reaction_diffusion"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "lstm_gated_reaction_diffusion"
    assert args.smooth_weight == 0.0
    assert args.align_weight == 0.7
    assert args.graph_state_dim == 64
    assert args.diffusion_mode == "deviation"
    assert args.output_dir == Path(
        "outputs/structured_baselines/lstm_gated_reaction_diffusion"
    )


def test_gru_z_gated_reaction_diffusion_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_gru_z_gated_reaction_diffusion"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "gru_z_gated_reaction_diffusion"
    assert args.smooth_weight == 0.0
    assert args.align_weight == 0.7
    assert args.graph_state_dim == 64
    assert args.diffusion_mode == "deviation"
    assert args.output_dir == Path(
        "outputs/structured_baselines/gru_z_gated_reaction_diffusion"
    )


def test_lstm_z_gated_reaction_diffusion_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_lstm_z_gated_reaction_diffusion"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "lstm_z_gated_reaction_diffusion"
    assert args.smooth_weight == 0.0
    assert args.align_weight == 0.7
    assert args.graph_state_dim == 64
    assert args.diffusion_mode == "deviation"
    assert args.output_dir == Path(
        "outputs/structured_baselines/lstm_z_gated_reaction_diffusion"
    )


def test_lstm_component_gated_reaction_diffusion_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=structured_baseline_lstm_component_gated_reaction_diffusion"
            ],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "lstm_component_gated_reaction_diffusion"
    assert args.component_scale_mode == "learnable"
    assert args.component_scale_init == 0.01
    assert args.smooth_weight == 0.0
    assert args.align_weight == 0.7
    assert args.output_dir == Path(
        "outputs/structured_baselines/lstm_component_gated_reaction_diffusion"
    )


def test_lstm_component_gated_no_scale_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment="
                "structured_baseline_lstm_component_gated_reaction_diffusion_no_scale"
            ],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "lstm_component_gated_reaction_diffusion"
    assert args.component_scale_mode == "none"
    assert args.component_scale_init == 0.01
    assert args.smooth_weight == 0.0
    assert args.align_weight == 0.7
    assert args.output_dir == Path(
        "outputs/structured_baselines/lstm_component_gated_reaction_diffusion_no_scale"
    )


def test_lstm_component_fusion_gated_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment="
                "structured_baseline_lstm_component_fusion_gated_reaction_diffusion"
            ],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "lstm_component_fusion_gated_reaction_diffusion"
    assert args.component_scale_mode == "learnable"
    assert args.component_scale_init == 0.01
    assert args.rd_fusion_gate_init == 0.1
    assert args.smooth_weight == 0.0
    assert args.align_weight == 0.7
    assert args.output_dir == Path(
        "outputs/structured_baselines/lstm_component_fusion_gated_reaction_diffusion"
    )


def test_lstm_component_fusion_gated_no_scale_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment="
                "structured_baseline_lstm_component_fusion_gated_reaction_diffusion_no_scale"
            ],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "lstm_component_fusion_gated_reaction_diffusion"
    assert args.component_scale_mode == "none"
    assert args.component_scale_init == 0.01
    assert args.rd_fusion_gate_init == 0.1
    assert args.smooth_weight == 0.0
    assert args.align_weight == 0.7
    assert args.output_dir == Path(
        "outputs/structured_baselines/"
        "lstm_component_fusion_gated_reaction_diffusion_no_scale"
    )


def test_stgcn_baseline_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_stgcn"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "stgcn"
    assert args.epochs == 200
    assert args.graph_state_dim == 64
    assert args.align_weight == 0.7
    assert args.align_temperature == 0.07
    assert args.align_dim == 64
    assert args.align_dropout == 0.3
    assert args.llm_forecast_csv.name == "forecast_events.csv"
    assert args.llm_embedding_npz.name == "forecast_event_embeddings.npz"
    assert args.ts_baseline_lookback == 5
    assert args.stgcn_kernel_size == 3
    assert args.output_dir == Path("outputs/structured_baselines/stgcn")


def test_agcrn_baseline_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_agcrn"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "agcrn"
    assert args.epochs == 200
    assert args.graph_state_dim == 64
    assert args.align_weight == 0.7
    assert args.align_temperature == 0.07
    assert args.align_dim == 64
    assert args.agcrn_embed_dim == 10
    assert args.llm_forecast_csv.name == "forecast_events.csv"
    assert args.llm_embedding_npz.name == "forecast_event_embeddings.npz"
    assert args.output_dir == Path("outputs/structured_baselines/agcrn")


def test_itransformer_baseline_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_itransformer"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "itransformer"
    assert args.epochs == 200
    assert args.graph_state_dim == 64
    assert args.align_weight == 0.0
    assert args.ts_baseline_lookback == 5
    assert args.itransformer_heads == 4
    assert args.output_dir == Path("outputs/structured_baselines/itransformer")


def test_timesnet_baseline_config_maps_to_structured_namespace() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=structured_baseline_timesnet"],
        )

    args = structured_namespace_from_cfg(cfg)

    assert args.propagation_model == "timesnet"
    assert args.epochs == 200
    assert args.graph_state_dim == 64
    assert args.align_weight == 0.0
    assert args.ts_baseline_lookback == 5
    assert args.timesnet_kernel_sizes == "1,3,5"
    assert args.output_dir == Path("outputs/structured_baselines/timesnet")
