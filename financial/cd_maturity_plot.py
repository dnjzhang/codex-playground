#!/usr/bin/env python3

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


MAX_EMPTY_MONTHS_BEFORE_BREAK = 3


def parse_dates(series: pd.Series, *, date_format: str | None) -> pd.Series:
    """Parse date strings, optionally falling back to auto-detection."""

    if date_format and date_format.lower() != "auto":
        parsed = pd.to_datetime(series, errors="coerce", format=date_format)
        if parsed.notna().any() or series.dropna().empty():
            if parsed.isna().any():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Could not infer format")
                    fallback = pd.to_datetime(series[parsed.isna()], errors="coerce")
                parsed = parsed.fillna(fallback)
            return parsed

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format")
        return pd.to_datetime(series, errors="coerce")


def build_broken_month_axis(
    monthly_totals: pd.DataFrame,
    *,
    max_empty_months: int = MAX_EMPTY_MONTHS_BEFORE_BREAK,
) -> tuple[list[pd.Timestamp | None], list[float]]:
    """Compress long zero-total gaps while preserving nearby empty months."""

    nonzero_positions = [
        idx
        for idx, amount in enumerate(monthly_totals["amount"])
        if amount > 0
    ]
    if not nonzero_positions:
        return list(monthly_totals.index), monthly_totals["amount"].tolist()

    display_months: list[pd.Timestamp | None] = [
        monthly_totals.index[nonzero_positions[0]]
    ]
    display_amounts: list[float] = [
        float(monthly_totals["amount"].iloc[nonzero_positions[0]])
    ]

    for current_pos, next_pos in zip(nonzero_positions, nonzero_positions[1:]):
        empty_positions = list(range(current_pos + 1, next_pos))

        if len(empty_positions) <= max_empty_months:
            kept_positions = empty_positions
            add_break = False
        else:
            kept_positions = empty_positions[:max_empty_months]
            add_break = True

        for pos in kept_positions:
            display_months.append(monthly_totals.index[pos])
            display_amounts.append(float(monthly_totals["amount"].iloc[pos]))

        if add_break:
            display_months.append(None)
            display_amounts.append(0.0)

        display_months.append(monthly_totals.index[next_pos])
        display_amounts.append(float(monthly_totals["amount"].iloc[next_pos]))

    return display_months, display_amounts


def plot_cd_maturities(csv_path: str, *, date_format: str | None = "%m/%d/%Y") -> Path | None:
    """Plot total CD amounts maturing each month.

    Improvements:
    - Ignore rows that do not start with a parseable date (e.g., headers, totals).
    - Keep the script executable via shebang so it can be run directly.
    - Allow overriding the incoming date format; defaults to U.S. month/day.
    """

    # Read only the first (date) and 5th (amount) columns as strings,
    # since some files may have headers, blanks, or subtotal rows.
    df = pd.read_csv(
        csv_path,
        header=None,
        usecols=[0, 4],
        names=["date_raw", "amount_raw"],
        dtype=str,
    )

    # Parse dates; non-date rows (headers, blanks) become NaT and are dropped.
    df["date"] = parse_dates(df["date_raw"], date_format=date_format)

    # Normalize amount text like "$22,000.00" -> 22000.0; drop unparseable.
    cleaned = df["amount_raw"].str.replace(r"[\$,]", "", regex=True)
    df["amount"] = pd.to_numeric(cleaned, errors="coerce")

    # Keep only valid date/amount rows
    df = df.dropna(subset=["date", "amount"]).copy()

    if df.empty:
        print("No valid date/amount rows found. Nothing to plot.")
        return None

    df.set_index("date", inplace=True)

    monthly_totals = df.resample("ME").sum()
    month_range = pd.date_range(
        start=monthly_totals.index.min(),
        end=monthly_totals.index.max(),
        freq="ME",
    )
    monthly_totals = monthly_totals.reindex(month_range, fill_value=0)

    display_months, display_amounts = build_broken_month_axis(monthly_totals)
    x_positions = list(range(len(display_months)))

    _, ax = plt.subplots()
    bars = ax.bar(x_positions, display_amounts)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total $ Matured (K)")
    ax.set_title(f"CD Maturity Totals per Month as of {date.today().isoformat()}")

    # Center labels inside each bar and skip zeros for readability
    labels = [
        f"${bar.get_height() / 1000:,.0f}K" if bar.get_height() > 0 else ""
        for bar in bars
    ]

    # Increase headroom so top labels don't collide with the border
    ax.margins(y=0.10)
    # Add one full bar-width of empty space at both ends
    n_bars = len(bars)
    if n_bars:
        ax.set_xlim(-1, n_bars)

    ax.bar_label(
        bars,
        labels=labels,
        label_type="edge",  # place labels at top edge of bars
        padding=3,
        color="black",
        fontsize=9,
        rotation=0,  # keep amount labels horizontal for readability
    )

    # Format y-axis in thousands (K)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"${x/1000:,.0f}K"))

    # Format x-axis as YYYY-MM for clarity; omit fake ticks for compressed gaps.
    date_tick_positions = [
        pos for pos, month in enumerate(display_months) if month is not None
    ]
    date_tick_labels = [
        month.strftime("%Y-%m") for month in display_months if month is not None
    ]
    ax.set_xticks(date_tick_positions, labels=date_tick_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Mark compressed timeline gaps directly on the x-axis baseline.
    axis_break_style = {
        "transform": ax.get_xaxis_transform(),
        "color": "black",
        "clip_on": False,
        "linewidth": 1.5,
        "solid_capstyle": "round",
    }
    for pos, month in enumerate(display_months):
        if month is None:
            ax.plot([pos - 0.20, pos - 0.05], [-0.02, 0.06], **axis_break_style)
            ax.plot([pos + 0.05, pos + 0.20], [-0.02, 0.06], **axis_break_style)

    plt.tight_layout()

    output_path = Path("/tmp") / f"{date.today().isoformat()}.png"
    ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot CD maturity totals per month")
    parser.add_argument("csv", help="Path to the CSV file with maturity dates and amounts")
    parser.add_argument(
        "--date-format",
        default="%m/%d/%Y",
        help=(
            "Optional strptime-style format for the date column. "
            'Use "auto" to let pandas infer (slower but flexible).'
        ),
    )
    args = parser.parse_args()
    output_path = plot_cd_maturities(args.csv, date_format=args.date_format)
    if output_path is not None:
        print(f"Saved chart to {output_path}")
