import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.scnews_sentiment.news.aggregate_daily_top_news import main


if __name__ == "__main__":
    raise SystemExit(main())
