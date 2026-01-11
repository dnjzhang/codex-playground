# Financial – CD Maturity Plot

Quick script to visualize total CD amounts maturing per month from a CSV.

## Run
- Create a venv (optional): `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install pandas matplotlib`
- Execute: `python financial/cd_maturity_plot.py financial/tracking-2025-09-07.csv`
  - CSV is parsed with columns 0 (date) and 4 (amount). Amounts may include $ and commas.

## Expected Output
- A bar chart window titled “CD Maturity Totals per Month” with month labels on X, dollar totals on Y, and values labeled on each bar.
- No files are written; close the window to finish.

## Tips
- Provide your own CSV in the same format: first column is a date, fifth column is the amount (e.g., `$10,000`).
- To save the figure instead of showing it, modify the script to call `plt.savefig('cd_maturity.png')` before `plt.show()`.
