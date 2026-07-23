from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def load_metrics(output_dir: Path) -> Dict[str, Any]:
    tscv_path = output_dir / "time_series_cv_summary.json"
    supervised_path = output_dir / "structured_supervised_metrics.json"
    metrics_path = tscv_path if tscv_path.exists() else supervised_path
    if not metrics_path.exists():
        raise FileNotFoundError(f"No structured metrics file found under {output_dir}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[int]:
    output_dir = Path(str(cfg.output_dir))
    metrics = load_metrics(output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
