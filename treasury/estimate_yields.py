import requests
from datetime import datetime, timedelta
import pandas as pd
import pandas_datareader.data as web  # pip install pandas_datareader

def get_upcoming_auctions(start_date_str: str, days: int) -> pd.DataFrame:
    # Same as in the previous response – retrieves upcoming auctions between the dates.
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = start_date + timedelta(days=days)
    filter_str = (
        f"auction_date:gte:{start_date},"
        f"auction_date:lte:{end_date}"
    )
    base_url = (
        "https://api.fiscaldata.treasury.gov/services/api/"
        "fiscal_service/v1/accounting/od/upcoming_auctions"
    )
    params = {"filter": filter_str, "page[size]": 100}
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.DataFrame(data)
    df = df.rename(columns={"announcemt_date": "announcement_date"})
    df["offering_amt"] = df["offering_amt"].replace({"null": None})
    return df

def estimate_yields(df: pd.DataFrame, fred_api_key: str) -> pd.DataFrame:
    """
    Estimate yields for each auction using FRED constant-maturity series.
    The function adds an 'estimated_yield' column (decimal, e.g., 0.0392 = 3.92%).
    """
    # Mapping from auction term to FRED series ID.
    # Bills: approximate using the closest maturity.  Notes/bonds use constant maturity.
    term_to_fred = {
        "4-Week": "TB1MS",      # 1-month Treasury Bill
        "6-Week": "TB3MS",      # use 3-month as a rough proxy
        "8-Week": "TB3MS",
        "13-Week": "TB3MS",
        "17-Week": "TB6MS",     # 6-month
        "26-Week": "TB6MS",
        "52-Week": "GS1",       # 1-year constant maturity (CMT)
        "2-Year": "GS2",
        "3-Year": "GS3",
        "5-Year": "GS5",
        "7-Year": "GS7",
        "10-Year": "GS10",
        "20-Year": "GS20",
        "30-Year": "GS30",
    }

    estimated = []
    # Determine date range for FRED look‑up – pull recent data (last 10 business days)
    today = datetime.today().date()
    start_fred = today - timedelta(days=14)

    for _, row in df.iterrows():
        term = row["security_term"]
        series = term_to_fred.get(term)
        if series is None:
            # Floating Rate Notes (FRNs) or unfamiliar terms: leave estimate blank.
            estimated.append(None)
            continue
        # Fetch the series from FRED; drop NaNs and take the latest value.
        try:
            fred_data = web.DataReader(series, "fred",
                                       start_fred, today,
                                       api_key=fred_api_key)
            latest = fred_data[series].dropna().iloc[-1]
            estimated.append(float(latest) / 100.0)  # convert percent to decimal
        except Exception:
            estimated.append(None)

    df["estimated_yield"] = estimated
    return df

if __name__ == "__main__":
    # Example usage:
    # 1. Query active auctions for the next 10 days starting 2025-07-28.
    auctions_df = get_upcoming_auctions("2025-07-28", 10)
    # 2. Estimate yields – supply your FRED API key here.
    fred_key = "YOUR_FRED_API_KEY"
    auctions_with_estimates = estimate_yields(auctions_df, fred_key)
    print(auctions_with_estimates[["auction_date", "security_type",
                                   "security_term", "estimated_yield"]])
    # 3. Save to CSV if desired:
    # auctions_with_estimates.to_csv("upcoming_auctions_with_estimates.csv", index=False)
