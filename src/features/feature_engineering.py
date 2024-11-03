# Create Lagged Returns:
# Calculate future returns over specific periods after the insider trade.

# Sort stock price data by Ticker and Date
stock_price_df.sort_values(by=['Ticker', 'Date'], inplace=True)

# Calculate daily returns
stock_price_df['Daily Return'] = stock_price_df.groupby('Ticker')['Close Price'].pct_change()

# Calculate cumulative returns over 5, 10, and 30 days
for days in [5, 10, 30]:
    stock_price_df[f'Return_{days}d'] = stock_price_df.groupby('Ticker')['Daily Return'].transform(lambda x: x.shift(-1).rolling(window=days).sum())

# Merge cumulative returns back into the final DataFrame
final_df = pd.merge(
    final_df,
    stock_price_df[['Ticker', 'Date', 'Return_5d', 'Return_10d', 'Return_30d']],
    left_on=['Ticker', 'Trade Date'],
    right_on=['Ticker', 'Date'],
    how='left'
)




# 2. Encode Categorical Variables: Convert categorical variables to numerical format.
# Transaction Type
final_df['TRANS_CODE_DESC'] = final_df['TRANS_CODE_DESC'].astype('category')
final_df['Transaction_Type_Code'] = final_df['TRANS_CODE_DESC'].cat.codes

# Insider Position
final_df['Insider Title'] = final_df['Insider Title'].astype('category')
final_df['Insider_Title_Code'] = final_df['Insider Title'].cat.codes


# 3. Create Additional Features: Generate features like moving averages and volatility measures.
# Moving Averages
for window in [5, 10, 20]:
    stock_price_df[f'MA_{window}'] = stock_price_df.groupby('Ticker')['Close Price'].transform(lambda x: x.rolling(window).mean())

# Merge moving averages back into the final DataFrame
final_df = pd.merge(
    final_df,
    stock_price_df[['Ticker', 'Date', f'MA_{window}']],
    left_on=['Ticker', 'Trade Date'],
    right_on=['Ticker', 'Date'],
    how='left'
)



# 4. Drop Unnecessary Columns: Remove columns that are not needed or that have been transformed.
columns_to_drop = ['Date_y', 'Date', 'TRANS_CODE', 'TRANS_CODE_DESC', 'Insider Title']
final_df.drop(columns=columns_to_drop, inplace=True)




# 5. Handle Missing Data Fill or Remove Missing Values:
# Check for missing values
missing_values = final_df.isnull().sum()
print(missing_values)

# Decide on a strategy: impute or drop
# For example, drop rows with missing target variable
final_df.dropna(subset=['Return_5d'], inplace=True)

# Fill missing numerical values with median
numerical_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
final_df[numerical_cols] = final_df[numerical_cols].fillna(final_df[numerical_cols].median())



# 6. Data Normalization and Scaling Normalize Financial Indicators:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
financial_cols = ['Revenue', 'Net Income', 'EPS', 'Total Assets']  # Add relevant columns
final_df[financial_cols] = scaler.fit_transform(final_df[financial_cols])



# 7. Save the cleaned and merged dataset for use in EDA and modeling.
processed_data_path = 'data/processed/'

# Ensure the directory exists
os.makedirs(processed_data_path, exist_ok=True)

# Save the final DataFrame
final_df.to_csv(os.path.join(processed_data_path, 'combined_data.csv'), index=False)
