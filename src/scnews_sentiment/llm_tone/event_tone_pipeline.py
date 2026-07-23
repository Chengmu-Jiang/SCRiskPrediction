"""Compatibility layer for the refactored LLM event-tone pipeline.

The implementation now lives in focused modules under
``src.scnews_sentiment.llm_tone``. This file keeps the historical import path
working for scripts and notebooks that import functions from
``event_tone_pipeline`` directly.
"""

from __future__ import annotations

from src.scnews_sentiment.llm_tone.cli import main, parse_args
from src.scnews_sentiment.llm_tone.data import *
from src.scnews_sentiment.llm_tone.eval import *
from src.scnews_sentiment.llm_tone.llm_clients import *
from src.scnews_sentiment.llm_tone.models import *
from src.scnews_sentiment.llm_tone.pipeline import *
from src.scnews_sentiment.llm_tone.prompts import *
from src.scnews_sentiment.llm_tone.train import *
from src.scnews_sentiment.llm_tone.utils import *


if __name__ == "__main__":
    main()
