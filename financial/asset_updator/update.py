"""Core update logic for asset_updator."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

from asset_updator.fetch_stooq import PriceQuote
from asset_updator.fetch_yfinance import DividendInfo
from asset_updator.io_csv import (
    format_date,
    format_number,
    parse_date,
    parse_float,
    parse_money,
)

LOGGER = logging.getLogger(__name__)

COLUMNS = [
    "Ticker",
    "Brokerage",
    "Purchase Date",
    "Purchase Price",
    "Initial Share Count",
    "Initial Value",
    "Current Share Count",
    "Current Price",
    "Current Value",
    "Gain/Loss",
    "Total Dividend",
    "Last Dividend",
    "Last Dividend Date",
    "Reinvest",
]


def is_reinvest(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().upper() == "Y"


def update_rows(
    rows: Iterable[dict],
    prices: Dict[str, PriceQuote],
    dividends: Dict[str, DividendInfo],
) -> List[dict]:
    updated: List[dict] = []

    for row in rows:
        updated_row = dict(row)
        ticker = str(updated_row.get("Ticker", "")).strip()
        if not ticker:
            updated.append(updated_row)
            continue

        price_quote = prices.get(ticker)
        dividend_info = dividends.get(ticker)
        current_price = price_quote.close if price_quote else None

        current_shares = parse_float(updated_row.get("Current Share Count"))
        if current_shares is None:
            current_shares = parse_float(updated_row.get("Initial Share Count"))
            if current_shares is not None:
                updated_row["Current Share Count"] = format_number(current_shares, 6)

        if current_price is None:
            updated_row["Current Price"] = ""
        else:
            updated_row["Current Price"] = format_number(current_price, 6)

        if dividend_info is not None:
            last_div_date = parse_date(updated_row.get("Last Dividend Date"))
            if last_div_date is None or dividend_info.date > last_div_date:
                updated_row["Last Dividend"] = format_number(dividend_info.amount, 6)
                updated_row["Last Dividend Date"] = format_date(dividend_info.date)

                if current_shares is None:
                    LOGGER.warning(
                        "Missing share count for %s; skipping dividend cash update",
                        ticker,
                    )
                else:
                    total_dividend = parse_money(updated_row.get("Total Dividend")) or 0.0
                    total_dividend += dividend_info.amount * current_shares
                    updated_row["Total Dividend"] = format_number(total_dividend, 2)

                if is_reinvest(updated_row.get("Reinvest")):
                    if current_shares is None:
                        LOGGER.warning(
                            "Missing share count for %s; skipping dividend reinvestment",
                            ticker,
                        )
                    elif current_price is None:
                        LOGGER.warning(
                            "Missing price for %s; skipping dividend reinvestment",
                            ticker,
                        )
                    else:
                        additional_shares = (
                            dividend_info.amount * current_shares
                        ) / current_price
                        current_shares += additional_shares
                        updated_row["Current Share Count"] = format_number(
                            current_shares, 6
                        )

        if current_price is not None and current_shares is not None:
            current_value = current_shares * current_price
            updated_row["Current Value"] = format_number(current_value, 2)
            initial_value = parse_money(updated_row.get("Initial Value"))
            if initial_value is not None:
                gain_loss = current_value - initial_value
                updated_row["Gain/Loss"] = format_number(gain_loss, 2)

        updated.append(updated_row)

    return updated
