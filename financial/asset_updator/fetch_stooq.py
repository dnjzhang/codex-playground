"""Fetch latest EOD prices from Stooq."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd
from pandas_datareader import data as pdr

LOGGER = logging.getLogger(__name__)
DEFAULT_LOOKBACK_DAYS = 14


@dataclass(frozen=True)
class PriceQuote:
    date: dt.date
    close: float


def normalize_stooq_symbol(ticker: str) -> str:
    text = ticker.strip()
    if not text:
        return text
    if "." in text:
        return text
    return f"{text}.US"


def fetch_latest_prices(
    tickers: Iterable[str],
    end_date: dt.date | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, PriceQuote]:
    if end_date is None:
        end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=lookback_days)
    results: Dict[str, PriceQuote] = {}

    normalized = {ticker.strip() for ticker in tickers if ticker and ticker.strip()}
    for ticker in sorted(normalized):
        stooq_symbol = normalize_stooq_symbol(ticker)
        try:
            data = pdr.DataReader(stooq_symbol, "stooq", start_date, end_date)
        except Exception as exc:
            LOGGER.warning(
                "Failed to fetch Stooq data for %s (symbol %s): %s",
                ticker,
                stooq_symbol,
                exc,
            )
            continue
        if data is None or data.empty:
            LOGGER.warning(
                "No Stooq price data for %s (symbol %s)", ticker, stooq_symbol
            )
            continue
        data = data.sort_index()
        last_row = data.iloc[-1]
        price_date = pd.Timestamp(data.index[-1]).date()
        price_close = float(last_row["Close"])
        results[ticker] = PriceQuote(date=price_date, close=price_close)

    return results
