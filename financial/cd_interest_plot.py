#!/usr/bin/env python3

from __future__ import annotations

import warnings
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


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


def plot_cd_interest(csv_path: str, *, date_format: str | None = "%m/%d/%Y") -> None:
    """Plot estimated CD interest earned each month across the portfolio."""

    # Read only the first (date), 5th (amount), and 6th (interest/yield) columns.
    df = pd.read_csv(
        csv_path,
        header=None,
        usecols=[0, 4, 5],
        names=["date_raw", "amount_raw", "rate_raw"],
        dtype=str,
    )

    df["date"] = parse_dates(df["date_raw"], date_format=date_format)

    cleaned_amount = df["amount_raw"].str.replace(r"[\$,]", "", regex=True)
    df["amount"] = pd.to_numeric(cleaned_amount, errors="coerce")

    cleaned_rate = df["rate_raw"].str.replace(r"[^0-9.\-]", "", regex=True)
    df["rate"] = pd.to_numeric(cleaned_rate, errors="coerce")

    df = df.dropna(subset=["date", "amount", "rate"]).copy()

    if df.empty:
        print("No valid date/amount/rate rows found. Nothing to plot.")
        return

    df["monthly_interest"] = df["amount"] * (df["rate"] / 100) / 12

    month_range = pd.period_range(
        start=df["date"].min().to_period("M"),
        end=df["date"].max().to_period("M"),
        freq="M",
    )
    monthly_interest = []
    for period in month_range:
        month_start = period.to_timestamp()
        total_interest = df.loc[df["date"] >= month_start, "monthly_interest"].sum()
        monthly_interest.append(total_interest)
    monthly_interest = pd.Series(monthly_interest, index=month_range)

    if monthly_interest.empty:
        print("No monthly interest totals found. Nothing to plot.")
        return

    def format_k(amount: float) -> str:
        return f"${amount / 1000:,.1f}K"

    cumulative_interest = monthly_interest.groupby(monthly_interest.index.year).cumsum()

    fig_width = max(10, len(monthly_interest) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x_positions = range(len(cumulative_interest))
    monthly_colors = ["tab:orange"] * len(monthly_interest)
    for _, group in monthly_interest.groupby(monthly_interest.index.year):
        if group.empty:
            continue
        max_idx = group.idxmax()
        min_idx = group.idxmin()
        max_pos = monthly_interest.index.get_loc(max_idx)
        min_pos = monthly_interest.index.get_loc(min_idx)
        if max_pos == min_pos:
            monthly_colors[max_pos] = "tab:green"
            continue
        monthly_colors[max_pos] = "tab:green"
        monthly_colors[min_pos] = "tab:red"
    cumulative_bars = ax.bar(
        x_positions,
        cumulative_interest.values,
        width=0.8,
        color="tab:blue",
        zorder=1,
    )
    ax.bar(
        x_positions,
        monthly_interest.values,
        width=0.35,
        color=monthly_colors,
        zorder=2,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Accumulated Interest ($)")
    ax.set_title(f"Accumulated CD Interest as of {date.today().isoformat()}")

    for idx, bar in enumerate(cumulative_bars):
        if bar.get_height() <= 0:
            continue
        month_value = monthly_interest.iloc[idx]
        running_total = cumulative_interest.iloc[idx]
        x_center = bar.get_x() + bar.get_width() / 2
        y_top = bar.get_height()
        ax.annotate(
            f"{format_k(running_total)}\n({format_k(month_value)})",
            (x_center, y_top),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            va="bottom",
            fontsize=8,
            linespacing=0.9,
            color="black",
        )

    ax.margins(y=0.10)
    n_bars = len(cumulative_bars)
    if n_bars:
        ax.set_xlim(-1, n_bars)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_k(x)))

    tick_positions = range(len(monthly_interest.index))
    tick_labels = [d.strftime("%Y-%m") for d in monthly_interest.index.to_timestamp("M")]
    ax.set_xticks(tick_positions, labels=tick_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot CD interest totals per month")
    parser.add_argument("csv", help="Path to the CSV file with maturity dates and rates")
    parser.add_argument(
        "--date-format",
        default="%m/%d/%Y",
        help=(
            "Optional strptime-style format for the date column. "
            'Use "auto" to let pandas infer (slower but flexible).'
        ),
    )
    args = parser.parse_args()
    plot_cd_interest(args.csv, date_format=args.date_format)
