# Stock Report Utility

Small CLI tool that pulls the latest available EOD close from Stooq and the
latest distribution (dividend) from Yahoo Finance (via yfinance), then writes
a CSV report.

## Requirements

- Python 3.11+
- Packages: `pandas`, `pandas-datareader`, `yfinance`

Install dependencies in your environment:

```bash
pip install pandas pandas-datareader yfinance
```

## Usage

Comma-separated tickers via CLI:

```bash
python stock_report.py --tickers AAPL,MSFT,SPY
```

Load tickers from a text file (commas or newlines are accepted):

```bash
python stock_report.py --file tickers.txt --output my_report.csv
```

The default output file name is `portfolio_report.csv`.

## Output Columns

- `ticker`
- `price_date`
- `price_close`
- `last_distribution_date`
- `last_distribution_per_share`

If a ticker has no price or distribution data, the row is still written with
blank fields and a warning is logged.
