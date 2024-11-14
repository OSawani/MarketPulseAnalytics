import os
import pandas as pd
from dataset_etl import run_data_pipeline

# --------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
print("shared variables script directory: ", current_dir)
if current_dir.endswith('src'):
    # go two directories up to get to the root directory
    os.chdir(os.path.join(current_dir, '..'))
    print("data_exploration app new current directory to: ", os.getcwd())
print("----------------------------\n")
# --------------------------------------------
run_data_pipeline()
# --------------------------------------------
# Paths
interim_insider_transactions_path = os.path.join('data', 'interim', 'insider_transactions', 'interim_insider_transactions.csv')
interim_stock_prices_path = os.path.join('data', 'interim', 'stock_prices', 'interim_stock_prices.csv')
interim_merged_path = os.path.join('data', 'interim', 'merged_insider_transactions_stock_prices', 'interim_merged_insider_transactions_stock_prices.csv')
processed_stock_prices_path = os.path.join('data', 'processed', 'stock_prices', 'processed_stock_prices.csv')

# DataFrames (loaded only once)
interim_df_insider_transactions = pd.read_csv(interim_insider_transactions_path)
interim_df_stock_prices = pd.read_csv(interim_stock_prices_path)
interim_df_merged = pd.read_csv(interim_merged_path)
processed_df_stock_prices = pd.read_csv(processed_stock_prices_path)

# Convert date columns to datetime
interim_df_merged['TRANS_DATE'] = pd.to_datetime(interim_df_merged['TRANS_DATE'])
interim_df_stock_prices['Date'] = pd.to_datetime(interim_df_stock_prices['Date'])
interim_df_insider_transactions['TRANS_DATE'] = pd.to_datetime(interim_df_insider_transactions['TRANS_DATE'])
processed_df_stock_prices['Date'] = pd.to_datetime(processed_df_stock_prices['Date'])
