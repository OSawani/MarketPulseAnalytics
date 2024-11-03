# Merge Insider Trading Data with Stock Price Data
# We'll merge the insider trading data with the stock price data on Ticker and align the dates.


# 1. Prepare Stock Price Data: Create a mapping of stock prices by date and ticker.
# Ensure both DataFrames have the same date format
merged_df['Trade Date'] = pd.to_datetime(merged_df['Trade Date']).dt.date
stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date']).dt.date

# Merge on Ticker and Date
combined_df = pd.merge(
    merged_df,
    stock_price_df,
    left_on=['Ticker', 'Trade Date'],
    right_on=['Ticker', 'Date'],
    how='left'
)


# 2. Handle Missing Merges: 
# Some trades might not have corresponding stock price data on the same date (e.g., weekends, holidays). 
# We will Consider merging with the next available trading day.
from pandas.tseries.offsets import BDay

def get_next_trading_day(date):
    next_day = date + BDay()
    return next_day.date()

# Attempt to fill missing stock prices by shifting to next trading day
missing_prices = combined_df['Close Price'].isnull()
combined_df.loc[missing_prices, 'Trade Date'] = combined_df.loc[missing_prices, 'Trade Date'].apply(get_next_trading_day)

# Retry merging
combined_df = pd.merge(
    combined_df.drop(columns=['Open Price', 'High Price', 'Low Price', 'Close Price', 'Adjusted Close', 'Trading Volume', 'Date']),
    stock_price_df,
    left_on=['Ticker', 'Trade Date'],
    right_on=['Ticker', 'Date'],
    how='left'
)



# 3. Merge with Financial Data: 
# Merge the combined DataFrame with the financial indicators on Ticker and Fiscal Year.
# Ensure 'Fiscal Year' is present in both DataFrames
combined_df['Fiscal Year'] = combined_df['Trade Date'].dt.year
financial_df['Fiscal Year'] = financial_df['Fiscal Year'].astype(int)

# Merge on Ticker and Fiscal Year
final_df = pd.merge(
    combined_df,
    financial_df,
    on=['Ticker', 'Fiscal Year'],
    how='left'
)
