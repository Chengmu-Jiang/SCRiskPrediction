from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from src.scnews_sentiment.structured.cli import parse_args


PATH_KEYS = {
    "news_csv",
    "label_csv",
    "nodes_csv",
    "edges_csv",
    "output_dir",
    "tensorboard_log_dir",
    "encoder_model",
    "checkpoint",
    "graph_checkpoint",
    "llm_forecast_csv",
    "llm_embedding_npz",
}


def _flatten_mapping(mapping: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = {}
    ignored_sections = {"hydra", "extras", "logger", "trainer", "paths"}

    def visit(value: Any, prefix: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if not prefix and key in ignored_sections:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                visit(item, path)
            else:
                flat[key] = path

    visit(mapping)
    # Top-level business keys should win over same-named fields inside helper groups
    # such as paths.output_dir.
    for key, item in mapping.items():
        if key in ignored_sections or isinstance(item, dict):
            continue
        flat[str(key)] = str(key)
    return flat


def structured_namespace_from_cfg(cfg: DictConfig) -> Namespace:
    """Convert a Hydra config into the namespace expected by the structured pipeline.

    The refactor keeps the original computational code nearly unchanged. This adapter is the
    boundary between Hydra's nested configuration style and the legacy flat argument names.
    """
    base = parse_args([])
    raw = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(raw, dict):
        raise TypeError("Hydra config must resolve to a mapping.")

    value_paths = _flatten_mapping(raw)
    allowed = set(vars(base))
    for key, cfg_path in value_paths.items():
        if key not in allowed:
            continue
        value = OmegaConf.select(cfg, cfg_path, default=None)
        if key in PATH_KEYS and value is not None:
            value = Path(str(value))
        setattr(base, key, value)

    if not hasattr(base, "split_override_masks"):
        base.split_override_masks = None
    return base
