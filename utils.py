def pull_data_from_yfinance(tickers, start_date, end_date):
    import yfinance as yf
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
        data[ticker] = data[ticker].rename_axis(columns=None).reset_index()
    return data