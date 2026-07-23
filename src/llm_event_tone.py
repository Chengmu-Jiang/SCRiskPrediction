import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.scnews_sentiment.llm_tone.event_tone_pipeline import main


if __name__ == "__main__":
    main()
