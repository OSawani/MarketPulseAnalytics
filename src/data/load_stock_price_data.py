# Load the stock price data from the Kaggle dataset "Huge Stock Market Dataset". 
# The data is organized in individual CSV files for each stock.


# Loading all stock files may be memory-intensive. 
# We are going to consider loading only the stocks present in your insider trading data.


# 1. Create a list of tickers from the insider trading data and load only those stock files.
# Get unique tickers from insider trading data
tickers = merged_df['Ticker'].unique()

# Load stock data only for relevant tickers
stock_dfs = []
for ticker in tickers:
    file_path = os.path.join(stock_data_path, f'{ticker}.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        df['Ticker'] = ticker
        stock_dfs.append(df)
    else:
        print(f"Stock data not found for ticker: {ticker}")

stock_price_df = pd.concat(stock_dfs, ignore_index=True)



# 2. Data Cleaning: Standardize Date Formats
# Convert Date column to datetime
stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])



# 3. Adjust Prices (If Necessary):
# Ensure that the stock prices are adjusted for splits and dividends. If not, we will consider using adjusted close prices.
# Check if 'Adj Close' is available
if 'Adj Close' in stock_price_df.columns:
    stock_price_df.rename(columns={'Adj Close': 'Adjusted Close'}, inplace=True)
else:
    # If not available, you may need to calculate it or fetch it from an API
    pass


# 4. Handle Missing Data:
# Check for missing values
missing_values = stock_price_df.isnull().sum()
print(missing_values)

# Drop rows with missing critical values
stock_price_df.dropna(subset=['Date', 'Close'], inplace=True)



# 5. Rename Columns for Consistency:
stock_price_df.rename(columns={
    'Open': 'Open Price',
    'High': 'High Price',
    'Low': 'Low Price',
    'Close': 'Close Price',
    'Volume': 'Trading Volume'
}, inplace=True)
