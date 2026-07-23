from typing import Optional

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.scnews_sentiment.config import structured_namespace_from_cfg
from src.scnews_sentiment.structured.pipeline import run_structured_pipeline


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[int]:
    args = structured_namespace_from_cfg(cfg)
    return run_structured_pipeline(args)


if __name__ == "__main__":
    main()
