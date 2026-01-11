# Asset Updator

Update a Google Sheets/Excel portfolio export with the latest prices and dividends.

## Setup
- Create a venv (optional): `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`

## Run (from repo root or `financial/`)
- `python -m asset_updator.cli --in portfolio.csv --out portfolio_updated.csv`
- `python -m asset_updator.cli --in portfolio.csv --inplace`
  - Creates a backup alongside the input file with a timestamp suffix.

## Example Input
```csv
Ticker,Brokerage,Purchase Date,Purchase Price,Initial Share Count,Initial Value,Current Share Count,Current Price,Current Value,Gain/Loss,Total Dividend,Last Dividend,Last Dividend Date,Reinvest
ABC,Test,01/01/2024,10,100,1000,,,,,,,
```

## Example Updated Columns
```csv
Ticker,Brokerage,Purchase Date,Purchase Price,Initial Share Count,Initial Value,Current Share Count,Current Price,Current Value,Gain/Loss,Total Dividend,Last Dividend,Last Dividend Date,Reinvest
ABC,Test,01/01/2024,10,100,1000,100,50,5000,4000,100,1,03/01/2024,
```

## Limitations
- Stooq may not have recent data for every ticker; missing prices skip value updates.
- yfinance dividends may be delayed; rate limiting is applied to reduce throttling.
- Dividend reinvestment requires a current price; otherwise cash dividends are still recorded.
