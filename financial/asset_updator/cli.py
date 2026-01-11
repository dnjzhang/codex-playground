"""CLI entry point for asset_updator."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
from pathlib import Path
from typing import Iterable

from asset_updator.fetch_stooq import fetch_latest_prices
from asset_updator.fetch_yfinance import fetch_latest_dividends
from asset_updator.io_csv import read_portfolio_csv, write_portfolio_csv
from asset_updator.update import update_rows

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update a portfolio CSV with latest prices and dividends."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Input portfolio CSV path.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        help="Output CSV path (ignored with --inplace).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input CSV in place.",
    )
    parser.add_argument(
        "--in-place",
        dest="inplace",
        action="store_true",
        help="Overwrite the input CSV in place.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inplace and args.output_path:
        raise SystemExit("--out is not allowed with --inplace")
    if not args.inplace and not args.output_path:
        raise SystemExit("Provide --out or use --inplace")

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input_path)
    output_path = input_path if args.inplace else Path(args.output_path)

    if args.inplace:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_path = Path(f"{input_path}-{timestamp}")
        shutil.copy2(input_path, backup_path)

    rows, fieldnames = read_portfolio_csv(str(input_path))
    tickers = [row.get("Ticker", "") for row in rows]

    prices = fetch_latest_prices(tickers)
    dividends = fetch_latest_dividends(tickers)

    updated_rows = update_rows(rows, prices, dividends)
    write_portfolio_csv(str(output_path), updated_rows, fieldnames)

    LOGGER.info("Updated %d rows", len(updated_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
