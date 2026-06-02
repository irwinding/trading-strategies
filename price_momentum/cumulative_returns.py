### A function that calculates the cumulative returns for a given stock over a specified time period. The function takes in the stock's price data and the time period (in months) as inputs and returns the cumulative return

def calculate_cumulative_return(price_data, months):
    """
    Calculate the cumulative return for a given stock over a specified time period.

    Parameters:
    price_data (pd.DataFrame): A DataFrame containing the stock's price data with a 'Close' column.
    months (int): The time period in months for which to calculate the cumulative return.

    Returns:
    float: The cumulative return over the specified time period.
    """
    # Calculate the number of trading days in the specified time period (assuming 21 trading days per month)
    trading_days = months * 21
    
    # Ensure there are enough data points to calculate the cumulative return
    if len(price_data) < trading_days:
        raise ValueError("Not enough data points to calculate cumulative return for the specified time period.")
    
    # Get the closing price at the start and end of the time period
    start_price = price_data['Close'].iloc[-trading_days]
    end_price = price_data['Close'].iloc[-1]
    
    # Calculate and return the cumulative return
    cumulative_return = (end_price - start_price) / start_price
    return cumulative_return

def calculate_cumulative_return_for_multiple_tickers(data, tickers, months):
    """
    Calculate the cumulative return for multiple stocks over a specified time period.

    Parameters:
    data (dict): A dictionary containing the price data for each stock.
    tickers (list): A list of stock tickers to calculate the cumulative return for.
    months (int): The time period in months for which to calculate the cumulative return.

    Returns:
    dict: A dictionary containing the cumulative return for each stock.
    """
    cumulative_returns = {}
    for ticker in tickers:
        cumulative_returns[ticker] = calculate_cumulative_return(data[ticker], months)
    
    # Sort the cumulative returns in descending order
    cumulative_returns = dict(sorted(cumulative_returns.items(), key=lambda item: item[1], reverse=True))
    
    return cumulative_returns