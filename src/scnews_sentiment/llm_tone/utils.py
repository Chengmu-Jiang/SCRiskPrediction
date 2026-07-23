from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_NEWS_CSV = Path("data") / "News" / "DailyTop_News" / "daily_top_news.csv"
DEFAULT_NODES_CSV = Path("data") / "IndustryChain" / "Filtered_Node_List_by_IGPC.csv"
DEFAULT_OUTPUT_DIR = Path("outputs") / "llm_event_tone"
DEFAULT_TOPIC_HS_MAP = {
    "Automaker": "8703",
    "Nexperia": "3818",
}
PATH_ARG_NAMES = {
    "news_csv",
    "nodes_csv",
    "output_dir",
    "forecast_prompt_path",
    "optimizer_prompt_path",
}


def normalize_config_key(key: str) -> str:
    return str(key).strip().lstrip("-").replace("-", "_")


def _parse_simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value[0] in {"'", '"'}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip("'\"")
    if value.startswith("[") or value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml_config(config_path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_container: Any = None

    with config_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))
            if indent == 0:
                if ":" not in stripped:
                    raise ValueError(f"Invalid YAML line {line_number} in {config_path}: {raw_line.rstrip()}")
                key, raw_value = stripped.split(":", 1)
                key = normalize_config_key(key)
                raw_value = raw_value.strip()
                if raw_value:
                    data[key] = _parse_simple_yaml_scalar(raw_value)
                    current_key = None
                    current_container = None
                else:
                    current_key = key
                    current_container = None
                    data[key] = None
                continue

            if current_key is None:
                raise ValueError(f"Unexpected indented YAML line {line_number} in {config_path}: {raw_line.rstrip()}")

            if stripped.startswith("- "):
                if current_container is None:
                    current_container = []
                    data[current_key] = current_container
                if not isinstance(current_container, list):
                    raise ValueError(f"Mixed YAML list/dict values for {current_key} in {config_path}")
                current_container.append(_parse_simple_yaml_scalar(stripped[2:].strip()))
            else:
                if ":" not in stripped:
                    raise ValueError(f"Invalid YAML line {line_number} in {config_path}: {raw_line.rstrip()}")
                if current_container is None:
                    current_container = {}
                    data[current_key] = current_container
                if not isinstance(current_container, dict):
                    raise ValueError(f"Mixed YAML list/dict values for {current_key} in {config_path}")
                item_key, raw_value = stripped.split(":", 1)
                current_container[str(item_key).strip()] = _parse_simple_yaml_scalar(raw_value.strip())

    return data


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        return dict(loaded or {})
    except ModuleNotFoundError:
        pass

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        return dict(loaded)
    except ModuleNotFoundError:
        return _load_simple_yaml_config(config_path)


def load_config_defaults(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    config_path = Path(config_path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml_config(config_path)
    else:
        raise ValueError(f"Unsupported config file type: {config_path}. Use .yaml, .yml, or .json.")
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping/object: {config_path}")

    defaults: Dict[str, Any] = {}
    for raw_key, value in data.items():
        key = normalize_config_key(raw_key)
        if key == "api_key":
            raise ValueError(
                "Do not put API keys in config files. Set an environment variable "
                "and configure api_key_env instead."
            )
        if key == "topic_hs_map" and isinstance(value, dict):
            value = [f"{topic}={code}" for topic, code in value.items()]
        defaults[key] = value
    return defaults


def coerce_config_arg_types(args: argparse.Namespace) -> argparse.Namespace:
    for name in PATH_ARG_NAMES:
        value = getattr(args, name, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    return args


def stable_hash(*parts: Any, length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def clean_generated_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = str(text)
    cleaned_text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned_text = re.sub(
        r"<think\b[^>]*>.*$",
        "",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned_text = re.sub(r"</think\s*>", "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text.strip()


def normalize_hs4(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        text = str(int(float(text)))
    except ValueError:
        text = re.sub(r"\D", "", text)
    return text.zfill(4)[-4:] if text else ""


def parse_topic_hs_map(items: Sequence[str]) -> Dict[str, str]:
    mapping = dict(DEFAULT_TOPIC_HS_MAP)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --topic-hs-map item: {item}. Use Topic=HS4.")
        topic, code = item.split("=", 1)
        mapping[topic.strip()] = normalize_hs4(code)
    return mapping


def parse_hidden_sizes(text: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("--hidden-sizes must contain at least one integer.")
    return values


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(device_arg: str) -> str:
    device_arg = str(device_arg).lower()
    import torch

    if device_arg in {"gpu", "cuda:gpu"}:
        device_arg = "cuda"
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Please run in a CUDA-enabled PyTorch environment or set the device to cpu/auto."
        )
    return device_arg


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    if len(y_true) == 0:
        return {"count": 0, "mse": None, "rmse": None, "mae": None, "r2": None}
    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(mse))
    if len(y_true) > 1 and float(np.var(y_true)) > 0:
        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = None
    return {"count": int(len(y_true)), "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def truncate_text(text: Any, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
