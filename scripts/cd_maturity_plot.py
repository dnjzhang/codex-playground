import pandas as pd
import matplotlib.pyplot as plt


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
    ax.set_ylabel("Total $ Matured")
    ax.set_title("CD Maturity Totals per Month")
    ax.bar_label(ax.containers[0], fmt="${:,.0f}", padding=3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot CD maturity totals per month")
    parser.add_argument("csv", help="Path to the CSV file with maturity dates and amounts")
    args = parser.parse_args()
    plot_cd_maturities(args.csv)
