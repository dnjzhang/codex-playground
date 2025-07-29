#!/usr/bin/env python
# coding: utf-8
from dotenv import load_dotenv

import requests
from datetime import datetime, timedelta
import pandas as pd

def get_upcoming_auctions(start_date_str: str, days: int) -> pd.DataFrame:
    """
    Query the FiscalData 'upcoming_auctions' dataset for auctions scheduled
    between start_date and start_date + days.

    Parameters
    ----------
    start_date_str : str
        Start of the date range in 'YYYY-MM-DD' format.
    days : int
        Number of days after start_date to include.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of auction records with fields:
        record_date, security_type, security_term, reopening, cusip,
        offering_amt, announcement_date, auction_date, issue_date.
    """
    # Parse start_date and compute end_date
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = start_date + timedelta(days=days)

    # Build the filter parameter for the API: auction_date between start and end
    filter_str = (
        f"auction_date:gte:{start_date},"
        f"auction_date:lte:{end_date}"
    )

    # Call the API
    base_url = (
        "https://api.fiscaldata.treasury.gov/services/api/"
        "fiscal_service/v1/accounting/od/upcoming_auctions"
    )
    params = {"filter": filter_str, "page[size]": 100}
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()  # raise an error if the request failed

    # Load JSON data into a DataFrame
    data = resp.json()["data"]
    df = pd.DataFrame(data)

    # Correct the announcement date field name and clean offering_amt
    df = df.rename(columns={"announcemt_date": "announcement_date"})
    df["offering_amt"] = df["offering_amt"].replace({"null": None})

    return df

def main():
    """Command line interface for getting upcoming Treasury auctions."""
    import argparse
    from datetime import datetime
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Query the FiscalData API for upcoming Treasury auctions.'
    )
    
    # Add arguments
    parser.add_argument(
        '--start-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Start date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to look ahead (default: 7)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (optional)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Get the auctions data
        df = get_upcoming_auctions(args.start_date, args.days)
        
        # Print the results
        print(f"\nFound {len(df)} upcoming auctions:")
        print(df)
        
        # Save to CSV if output path is provided
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
            
    except ValueError as e:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD format.")
        return 1
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch data from API: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())