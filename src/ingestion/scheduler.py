"""
Scheduler for automated weather data ingestion.

Runs the ingestion pipeline on a configurable interval.
In production (EC2), this is replaced by cron jobs — but this
scheduler works for local development and Docker.

Usage:
    python -m src.ingestion.scheduler                    # Run once
    python -m src.ingestion.scheduler --continuous       # Run every 6 hours
    python -m src.ingestion.scheduler --backfill 14      # Backfill 14 days then exit
"""

import sys
import time

import schedule

from src.config import INGESTION_INTERVAL_HOURS, BACKFILL_DAYS
from src.ingestion.weather_fetcher import run_ingestion
from src.utils.logger import logger


def run_scheduled_ingestion():
    """Wrapper for the scheduler to call."""
    try:
        stats = run_ingestion(past_days=1, forecast_days=2)
        logger.info(f"Scheduled ingestion result: {stats['status']}")
    except Exception as e:
        logger.error(f"Scheduled ingestion failed: {e}")


def run_continuous(interval_hours: int = None):
    """
    Run ingestion on a recurring schedule.

    Args:
        interval_hours: Hours between ingestion runs (default from config)
    """
    if interval_hours is None:
        interval_hours = INGESTION_INTERVAL_HOURS

    logger.info(f"Starting continuous ingestion scheduler (every {interval_hours} hours)")

    # Run once immediately
    run_scheduled_ingestion()

    # Schedule recurring runs
    schedule.every(interval_hours).hours.do(run_scheduled_ingestion)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    backfill = BACKFILL_DAYS

    # Parse CLI arguments
    for i, arg in enumerate(sys.argv):
        if arg == "--backfill" and i + 1 < len(sys.argv):
            backfill = int(sys.argv[i + 1])

    if "--continuous" in sys.argv:
        run_continuous()
    else:
        # Single run (default)
        logger.info(f"Running single ingestion (backfill={backfill} days)")
        stats = run_ingestion(past_days=backfill)
        print(f"\nIngestion stats: {stats}")