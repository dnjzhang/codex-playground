# Treasury – Auction Queries and Yield Estimates

Tools for querying upcoming Treasury auctions and estimating yields using FRED data.

## Query Upcoming Auctions
- Install: `pip install pandas requests python-dotenv`
- Run: `python treasury/query_treasury_auctions.py --start-date 2025-07-28 --days 10 --output auctions.csv`
- Expected: prints the number of auctions and a table; writes `auctions.csv` if `--output` is given.

## Estimate Yields
- Extra dependency: `pip install pandas_datareader`
- Edit `treasury/estimate_yields.py` to set `fred_key = "YOUR_FRED_API_KEY"` (or pass into your own wrapper).
- Run example inside the script or import and call `estimate_yields(df, fred_key)`.
- Expected: prints columns `auction_date`, `security_type`, `security_term`, `estimated_yield` (decimal, e.g., 0.0392).

## Notes
- Requires internet access to Treasury FiscalData API and FRED.
- `auction_query.sh` is a simple curl-based example; see the script for parameters.

