"""Fetch latest dividends from yfinance with retries."""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)
DEFAULT_PAUSE_SECONDS = 0.4
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 0.5


@dataclass(frozen=True)
class DividendInfo:
    date: dt.date
    amount: float


def normalize_yfinance_symbol(ticker: str) -> str:
    text = ticker.strip()
    if text.upper().endswith(".US"):
        return text[: -len(".US")]
    return text


def is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    tokens = [
        "429",
        "too many requests",
        "timed out",
        "timeout",
        "temporarily",
        "connection",
        "network",
        "502",
        "503",
        "504",
    ]
    return any(token in message for token in tokens)


def _fetch_latest_dividend_for_symbol(
    yf_symbol: str,
    max_retries: int,
    backoff_seconds: float,
) -> Optional[DividendInfo]:
    for attempt in range(max_retries):
        if attempt > 0:
            delay = backoff_seconds * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            dividends = yf.Ticker(yf_symbol).dividends
            if dividends is None or dividends.empty:
                return None
            dividends = dividends.dropna().sort_index()
            if dividends.empty:
                return None
            last_date = pd.Timestamp(dividends.index[-1]).date()
            last_amount = float(dividends.iloc[-1])
            return DividendInfo(date=last_date, amount=last_amount)
        except Exception as exc:
            if not is_retryable_error(exc) or attempt == max_retries - 1:
                LOGGER.warning(
                    "yfinance failed for %s (attempt %d/%d): %s",
                    yf_symbol,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                return None
            LOGGER.warning(
                "yfinance retryable error for %s (attempt %d/%d): %s",
                yf_symbol,
                attempt + 1,
                max_retries,
                exc,
            )
    return None


def fetch_latest_dividends(
    tickers: Iterable[str],
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> Dict[str, DividendInfo]:
    results: Dict[str, DividendInfo] = {}
    normalized = {ticker.strip() for ticker in tickers if ticker and ticker.strip()}
    for ticker in sorted(normalized):
        yf_symbol = normalize_yfinance_symbol(ticker)
        info = _fetch_latest_dividend_for_symbol(yf_symbol, max_retries, backoff_seconds)
        if info:
            results[ticker] = info
        time.sleep(pause_seconds)
    return results
