import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.scnews_sentiment.tuning.structured_pipeline_tuner import main


if __name__ == "__main__":
    raise SystemExit(main())
