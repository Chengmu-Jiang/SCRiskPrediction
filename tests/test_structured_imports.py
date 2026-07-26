def test_structured_modules_import() -> None:
    import pytest

    pytest.importorskip("numpy")
    pytest.importorskip("pandas")

    import src.scnews_sentiment.structured.common  # noqa: F401
    import src.scnews_sentiment.structured.data  # noqa: F401
    import src.scnews_sentiment.structured.models  # noqa: F401
    import src.scnews_sentiment.structured.pipeline  # noqa: F401

    pytest.importorskip("torch")
    import src.scnews_sentiment.structured.propagation_models  # noqa: F401
    import src.scnews_sentiment.structured.training_utils  # noqa: F401


def test_stgcn_uses_structured_training_branch() -> None:
    import pytest

    pytest.importorskip("torch")
    from src.scnews_sentiment.structured.propagation_models import (
        is_label_history_propagation_model,
    )

    assert not is_label_history_propagation_model("stgcn")
    assert not is_label_history_propagation_model("agcrn")
    assert is_label_history_propagation_model("itransformer")
    assert is_label_history_propagation_model("timesnet")


def test_rd_fusion_gate_init_sets_constant_initial_probability() -> None:
    import pytest

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from src.scnews_sentiment.structured.propagation_models import (
        LSTMComponentFusionGatedReactionDiffusionPropagationNetwork,
    )

    model = LSTMComponentFusionGatedReactionDiffusionPropagationNetwork(
        node_igpc=np.ones(3, dtype=np.float32),
        normalized_adjacency=np.eye(3, dtype=np.float32),
        risk_dim=2,
        state_dim=4,
        risk_hidden_sizes=(),
        recovery_hidden_sizes=(),
        dropout=0.0,
        dt=1.0,
        alpha_init=0.1,
        beta_init=0.1,
        gamma_init=0.1,
        eta_init=0.1,
        eps=1e-6,
        diffusion_mode="deviation",
        initial_state_mode="igpc",
        enable_graph_diffusion=True,
        enable_two_hop_diffusion=True,
        enable_recovery=True,
        component_scale_mode="learnable",
        component_scale_init=0.3,
        rd_fusion_gate_init=0.9,
    )

    assert torch.allclose(
        model.rd_fusion_gate.weight,
        torch.zeros_like(model.rd_fusion_gate.weight),
    )
    assert torch.allclose(
        torch.sigmoid(model.rd_fusion_gate.bias),
        torch.full_like(model.rd_fusion_gate.bias, 0.9),
        atol=1e-6,
    )


def test_grand_t_and_gread_t_forward_are_finite() -> None:
    import pytest

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from src.scnews_sentiment.structured.propagation_models import (
        GRANDTPropagationNetwork,
        GREADTPropagationNetwork,
    )

    adjacency = np.asarray(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    common = {
        "node_igpc": np.ones(3, dtype=np.float32),
        "normalized_adjacency": adjacency,
        "risk_dim": 5,
        "state_dim": 8,
        "risk_hidden_sizes": (),
        "dropout": 0.0,
        "initial_state_mode": "fixed_zero",
        "ode_steps": 2,
        "ode_step_size": 0.25,
    }
    grand = GRANDTPropagationNetwork(
        **common,
        heads=2,
        attention_dim=4,
    )
    gread = GREADTPropagationNetwork(
        **common,
        reaction="bs",
        use_soft_adjacency=True,
        alpha_init=0.1,
        beta_init=0.1,
    )
    inputs = torch.randn(4, 3, 5)
    mask = torch.ones(4, 3, 1)

    grand_states = grand(inputs, mask)
    gread_states = gread(inputs, mask)

    assert grand_states.shape == (4, 3, 8)
    assert gread_states.shape == (4, 3, 8)
    assert torch.isfinite(grand_states).all()
    assert torch.isfinite(gread_states).all()
    grand_states.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in grand.parameters()
    )
    assert torch.allclose(
        grand.attention_matrix(grand_states[-1]).sum(dim=1),
        torch.ones(3),
        atol=1e-6,
    )


def test_prospective_mlp_uses_embedding_presence_mask() -> None:
    import pytest

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from src.scnews_sentiment.structured.propagation_models import (
        ProspectiveMLPPropagationNetwork,
        is_prospective_semantic_model,
    )

    model = ProspectiveMLPPropagationNetwork(
        normalized_adjacency=np.eye(3, dtype=np.float32),
        risk_dim=6,
        state_dim=4,
        dropout=0.0,
    )
    embeddings = torch.randn(2, 3, 6)
    mask = torch.tensor(
        [[[1.0], [0.0], [1.0]], [[0.0], [1.0], [1.0]]]
    )
    states = model(embeddings, mask)

    assert is_prospective_semantic_model("prospective_mlp")
    assert states.shape == (2, 3, 4)
    assert torch.isfinite(states).all()
    assert torch.count_nonzero(model.normalized_adjacency) == 0


def test_repeated_cv_summary_aggregates_repeat_level_values() -> None:
    import pytest

    pd = pytest.importorskip("pandas")
    from src.scnews_sentiment.structured.pipeline import (
        _parse_repeat_seeds,
        _summarize_repeated_cv_fold_metrics,
        _summarize_repeated_cv_metrics,
    )

    assert _parse_repeat_seeds("42,2026,3407") == [42, 2026, 3407]

    repeat_metrics = pd.DataFrame(
        [
            {"repeat": 0, "seed": 42, "split": "test", "metric": "rmse", "value": 2.0},
            {"repeat": 1, "seed": 2026, "split": "test", "metric": "rmse", "value": 4.0},
            {"repeat": 2, "seed": 3407, "split": "test", "metric": "rmse", "value": 6.0},
        ]
    )
    aggregate = _summarize_repeated_cv_metrics(repeat_metrics)
    row = aggregate.loc[
        (aggregate["split"] == "test") & (aggregate["metric"] == "rmse")
    ].iloc[0]

    assert row["count"] == 3
    assert row["mean"] == pytest.approx(4.0)
    assert row["std"] == pytest.approx(2.0)

    fold_summary = pd.DataFrame(
        [
            {"repeat": 0, "fold": 0, "val_rmse": 1.0, "test_rmse": 2.0},
            {"repeat": 0, "fold": 1, "val_rmse": 3.0, "test_rmse": 4.0},
            {"repeat": 1, "fold": 0, "val_rmse": 5.0, "test_rmse": 6.0},
            {"repeat": 1, "fold": 1, "val_rmse": 7.0, "test_rmse": 8.0},
        ]
    )
    fold_aggregate = _summarize_repeated_cv_fold_metrics(fold_summary)
    fold_row = fold_aggregate.loc[
        (fold_aggregate["split"] == "test") & (fold_aggregate["metric"] == "rmse")
    ].iloc[0]

    assert fold_row["count"] == 4
    assert fold_row["mean"] == pytest.approx(5.0)
    assert fold_row["std"] == pytest.approx(2.581988897471611)
