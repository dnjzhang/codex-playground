import requests
import pandas as pd

def fetch_cd_rates(fred_api_key: str) -> pd.DataFrame:
    """
    Fetch the latest national deposit rates for selected CD maturities from FRED.

    Parameters
    ----------
    fred_api_key : str
        Your FRED API key (register at fred.stlouisfed.org to obtain one).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns: maturity, series_id, date, rate (percent).
    """
    # FRED series IDs for non-callable CD rates (<100M deposits)
    cd_series = {
        "3-month": "TY3MCD",
        "6-month": "TY6MCD",
        "12-month": "TY12MCD",
        "18-month": "TY18MCD",
        "24-month": "TY24MCD",  # 2 years
        "36-month": "TY36MCD",  # 3 years
    }

    results = []
    for maturity, series_id in cd_series.items():
        # Build API request for the latest observation
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": fred_api_key,
            "file_type": "json",
            "sort_order": "desc",  # return newest observations first
            "limit": 1             # retrieve only the most recent value
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data["observations"]:
            obs = data["observations"][0]
            rate_value = None
            # FRED sometimes uses "." for missing values; convert to float if possible
            try:
                rate_value = float(obs["value"])
            except ValueError:
                rate_value = None

            results.append({
                "maturity": maturity,
                "series_id": series_id,
                "date": obs["date"],
                "rate": rate_value  # expressed in percent (e.g., 4.35 for 4.35%)
            })
        else:
            results.append({
                "maturity": maturity,
                "series_id": series_id,
                "date": None,
                "rate": None
            })

    return pd.DataFrame(results)

# Example usage:
if __name__ == "__main__":
    fred_key = "YOUR_FRED_API_KEY"
    cd_rates_df = fetch_cd_rates(fred_key)
    print(cd_rates_df)
    # cd_rates_df.to_csv("prevailing_cd_rates.csv", index=False)
