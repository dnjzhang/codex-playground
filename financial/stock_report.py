#!/usr/bin/env python3
"""Generate a CSV report with latest EOD prices and distributions for US tickers."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT = "portfolio_report.csv"
YFINANCE_PAUSE_SECONDS = 0.4
YFINANCE_MAX_RETRIES = 3
YFINANCE_BACKOFF_SECONDS = 0.5


def parse_ticker_string(value: str) -> List[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",")
    parts = [part.strip() for part in normalized.split(",")]
    return [part for part in parts if part]


def load_tickers_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    return parse_ticker_string(content)


def normalize_stooq_symbol(ticker: str) -> str:
    if ticker.upper().endswith(".US"):
        return ticker
    return f"{ticker}.US"


def normalize_yfinance_symbol(ticker: str) -> str:
    if ticker.upper().endswith(".US"):
        return ticker[: -len(".US")]
    return ticker


def fetch_latest_close(
    stooq_symbol: str, end_date: dt.date
) -> Tuple[Optional[str], Optional[float]]:
    start_date = end_date - dt.timedelta(days=14)
    data = pdr.DataReader(stooq_symbol, "stooq", start_date, end_date)
    if data.empty:
        return None, None
    data = data.sort_index()
    last_row = data.iloc[-1]
    price_date = pd.Timestamp(data.index[-1]).date().isoformat()
    price_close = float(last_row["Close"])
    return price_date, price_close


def fetch_latest_distribution(
    yf_symbol: str,
    max_retries: int = YFINANCE_MAX_RETRIES,
    backoff_seconds: float = YFINANCE_BACKOFF_SECONDS,
) -> Tuple[Optional[str], Optional[float]]:
    for attempt in range(max_retries):
        if attempt > 0:
            delay = backoff_seconds * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            dividends = yf.Ticker(yf_symbol).dividends
            if dividends is None or dividends.empty:
                return None, None
            dividends = dividends.dropna()
            if dividends.empty:
                return None, None
            last_value = float(dividends.iloc[-1])
            last_date = pd.Timestamp(dividends.index[-1]).date().isoformat()
            return last_date, last_value
        except Exception as exc:
            LOGGER.warning(
                "yfinance failed for %s (attempt %d/%d): %s",
                yf_symbol,
                attempt + 1,
                max_retries,
                exc,
            )
    return None, None


def format_float(value: Optional[float], decimals: int) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def format_date(value: Optional[str]) -> str:
    return value or ""


def build_rows(tickers: Iterable[str]) -> List[dict]:
    rows: List[dict] = []
    end_date = dt.date.today()
    for ticker in tickers:
        price_date = None
        price_close = None
        distribution_date = None
        distribution_amount = None

        stooq_symbol = normalize_stooq_symbol(ticker)
        yf_symbol = normalize_yfinance_symbol(ticker)

        try:
            price_date, price_close = fetch_latest_close(stooq_symbol, end_date)
            if price_date is None:
                LOGGER.warning(
                    "No Stooq price data for %s (symbol %s)", ticker, stooq_symbol
                )
        except Exception as exc:
            LOGGER.warning(
                "Failed to fetch Stooq data for %s (symbol %s): %s",
                ticker,
                stooq_symbol,
                exc,
            )

        distribution_date, distribution_amount = fetch_latest_distribution(yf_symbol)
        if distribution_date is None:
            LOGGER.warning(
                "No yfinance distribution data for %s (symbol %s)",
                ticker,
                yf_symbol,
            )

        rows.append(
            {
                "ticker": ticker,
                "price_date": format_date(price_date),
                "price_close": format_float(price_close, 4),
                "last_distribution_date": format_date(distribution_date),
                "last_distribution_per_share": format_float(distribution_amount, 6),
            }
        )

        time.sleep(YFINANCE_PAUSE_SECONDS)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CSV report with latest EOD prices and distributions."
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated list of tickers (e.g., AAPL,MSFT).",
    )
    parser.add_argument(
        "--file",
        help="Path to a text file with tickers separated by commas or newlines.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    tickers: List[str] = []
    if args.tickers:
        tickers.extend(parse_ticker_string(args.tickers))
    if args.file:
        tickers.extend(load_tickers_from_file(args.file))

    if not tickers:
        raise SystemExit("Error: provide tickers via --tickers or --file.")

    LOGGER.info("Fetching data for %d ticker(s)...", len(tickers))
    rows = build_rows(tickers)

    columns = [
        "ticker",
        "price_date",
        "price_close",
        "last_distribution_date",
        "last_distribution_per_share",
    ]
    report = pd.DataFrame(rows, columns=columns)
    report.to_csv(args.output, index=False)

    LOGGER.info("Wrote report to %s", args.output)


if __name__ == "__main__":
    main()
