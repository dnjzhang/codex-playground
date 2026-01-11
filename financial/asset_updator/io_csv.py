"""CSV I/O and parsing helpers for asset_updator."""

from __future__ import annotations

import csv
import datetime as dt
from typing import Iterable, List, Optional, Sequence

from dateutil import parser as date_parser


def parse_money(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace("$", "").replace(",", "")
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_float(value: Optional[str]) -> Optional[float]:
    return parse_money(value)


def parse_date(value: Optional[str]) -> Optional[dt.date]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        try:
            return dt.datetime.strptime(text, "%m/%d/%Y").date()
        except ValueError:
            pass
    try:
        parsed = date_parser.parse(text, dayfirst=False, yearfirst=True)
        return parsed.date()
    except (ValueError, TypeError, OverflowError):
        return None


def format_date(value: Optional[dt.date]) -> str:
    if value is None:
        return ""
    return value.strftime("%m/%d/%Y")


def format_number(value: Optional[float], decimals: int = 6) -> str:
    if value is None:
        return ""
    formatted = f"{value:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def read_portfolio_csv(path: str) -> tuple[List[dict], Sequence[str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def write_portfolio_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
