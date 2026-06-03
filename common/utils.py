def pull_data_from_yfinance(tickers, start_date, end_date):
    import yfinance as yf
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
        data[ticker] = data[ticker].rename_axis(columns=None).reset_index()
    return data


def fetch_risk_free_rate(as_of_date=None):
    """Return the annualised US 3-month T-bill yield (decimal, not percent).

    Parameters
    ----------
    as_of_date:
        ISO date string (e.g. '2024-12-31'). Fetches the yield on or just
        before this date. Defaults to the most recent available value.
    """
    import yfinance as yf
    import pandas as pd

    anchor = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today()
    start = (anchor - pd.DateOffset(days=7)).strftime("%Y-%m-%d")
    end = anchor.strftime("%Y-%m-%d")

    data = yf.download("^IRX", start=start, end=end, progress=False, multi_level_index=False)
    if data.empty:
        raise ValueError(f"Could not fetch ^IRX data around {as_of_date or 'today'}.")
    # ^IRX is quoted in percent (e.g. 5.25 means 5.25%)
    return float(data["Close"].dropna().iloc[-1]) / 100