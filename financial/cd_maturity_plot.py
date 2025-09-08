import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import date


def plot_cd_maturities(csv_path: str) -> None:
    """Plot total CD amounts maturing each month."""
    df = pd.read_csv(csv_path, header=None, usecols=[0, 4], names=["date", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].replace(r"[\$,]", "", regex=True).astype(float)
    df.set_index("date", inplace=True)

    monthly_totals = df.resample("M").sum()
    month_range = pd.date_range(start=monthly_totals.index.min(), end=monthly_totals.index.max(), freq="M")
    monthly_totals = monthly_totals.reindex(month_range, fill_value=0)

    ax = monthly_totals.plot.bar(legend=False)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total $ Matured (K)")
    ax.set_title(f"CD Maturity Totals per Month as of {date.today().isoformat()}")

    # Center labels inside each bar and skip zeros for readability
    bars = ax.containers[0]
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

    # Format x-axis as YYYY-MM for clarity and keep labels centered on bars
    ax.set_xticklabels([d.strftime("%Y-%m") for d in monthly_totals.index], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot CD maturity totals per month")
    parser.add_argument("csv", help="Path to the CSV file with maturity dates and amounts")
    args = parser.parse_args()
    plot_cd_maturities(args.csv)
