import os
import pandas as pd
import numpy as np
import re
import zipfile
import shutil
import requests

def download_zip_file(url, download_path, filename):
    file_path = os.path.join(download_path, filename)
    
    # Check if the file already exists
    if not os.path.exists(file_path):
        # If the file doesn't exist, download it
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Check if the request was successful
            total_size = int(r.headers.get('Content-Length', 0))  # Get the total file size
            
            with open(file_path, 'wb') as f:
                downloaded_size = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Calculate the progress percentage
                        progress = (downloaded_size / total_size) * 100 if total_size else 0
                        print(f"\rDownloading... {progress:.2f}% complete", end="")

        print(f"\nFile downloaded successfully and saved as {file_path}")
        print("----------------------------\n")
    else:
        # If the file exists, print a message
        print(f"File already exists: {file_path}")
        print("----------------------------\n")



def unzip_stock_prices(zip_path, unzip_path, specific_folder='Stocks'):
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist. Please download the zip file first.")
        return

    # Get the directory of the zip file to extract the Stocks folder in the same location
    zip_dir = os.path.dirname(zip_path)
    
    print(f"Unzipping {zip_path} to {zip_dir}...")

    # Check if the destination folder is empty (excluding .gitkeep files)
    if os.path.exists(unzip_path):
        files_in_unzip_path = os.listdir(unzip_path)
        if files_in_unzip_path != ['.gitkeep']:  # If it's not empty or only contains .gitkeep
            print(f"The destination folder '{unzip_path}' is not empty. Aborting unzipping.")
            print("----------------------------\n")
            return

    # Extract the Stocks folder to the same location as the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all files in the zip
        all_files = zip_ref.namelist()

        # Filter for files that start with the specific folder name (e.g., 'Stocks/')
        files_to_extract = [file for file in all_files if file.startswith(specific_folder)]
        
        # Total number of files to extract
        total_files = len(files_to_extract)

        # Extract the files with progress indicator
        for idx, file in enumerate(files_to_extract):
            zip_ref.extract(file, zip_dir)
            # Calculate and display the progress percentage
            progress = (idx + 1) / total_files * 100
            print(f"\rProgress: {progress:.2f}%", end="")  # Update the same line in the terminal

    # Path to the extracted Stocks folder
    stocks_folder_path = os.path.join(zip_dir, specific_folder)

    # Ensure the Stocks folder exists before moving its contents
    if os.path.exists(stocks_folder_path):
        # Move all contents of the Stocks folder to the target directory
        for item in os.listdir(stocks_folder_path):
            source = os.path.join(stocks_folder_path, item)
            destination = os.path.join(unzip_path, item)
            
            # Move files or directories
            if os.path.isdir(source):
                shutil.move(source, destination)
            else:
                shutil.move(source, destination)
        
        # Delete the Stocks folder after moving its contents
        shutil.rmtree(stocks_folder_path)
        
        print(f"\nMoved contents from '{stocks_folder_path}' to '{unzip_path}' and deleted the folder.")
    else:
        print(f"Error: The folder '{specific_folder}' was not found in the zip file.")

    print(f"\nUnzipping complete. Files extracted and moved to {unzip_path}.")
    print("----------------------------\n")



def unzip_insider_transactions(zip_path, unzip_path):
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist. Please download the zip file first.")
        return

    print(f"Unzipping {zip_path} to {unzip_path}...")

    # Check if the destination folder is empty (excluding .gitkeep files)
    if os.path.exists(unzip_path):
        files_in_unzip_path = os.listdir(unzip_path)
        if files_in_unzip_path != ['.gitkeep']:  # If it's not empty or only contains .gitkeep
            print(f"The destination folder '{unzip_path}' is not empty. Aborting unzipping.")
            print("----------------------------\n")
            return

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Total number of files to extract
        total_files = len(zip_ref.namelist())

        # Extract all files with progress indicator
        for idx, file in enumerate(zip_ref.namelist()):
            zip_ref.extract(file, unzip_path)
            # Calculate and display the progress percentage
            progress = (idx + 1) / total_files * 100
            print(f"\rProgress: {progress:.2f}%", end="")  # Update the same line in the terminal

    print(f"\nUnzipping complete. Files extracted to {unzip_path}.")
    print("----------------------------\n")


def process_insider_transactions():
    files = [os.path.join('data', 'raw', 'insider_transactions', f'{year}q{quarter}_form345', 'NONDERIV_TRANS.tsv')
             for year in range(2014, 2018) for quarter in range(1, 5)]
    dataframes = []
    for file in files:
        if os.path.exists(file):
            try:
                temp = pd.read_csv(file, sep='\t', low_memory=False)
                dataframes.append(temp)
            except Exception as e:
                ...
                # print(f'Error reading {file}: {e}')
        else:
            print(f'File {file} does not exist')
    df = pd.concat(dataframes, ignore_index=True)

    columns_to_drop = ['DIRECT_INDIRECT_OWNERSHIP_FN',
                       'NATURE_OF_OWNERSHIP',
                       'NATURE_OF_OWNERSHIP_FN',
                       'VALU_OWND_FOLWNG_TRANS',
                       'VALU_OWND_FOLWNG_TRANS_FN',                   
                       'SHRS_OWND_FOLWNG_TRANS_FN',
                       'TRANS_ACQUIRED_DISP_CD_FN',
                       'TRANS_PRICEPERSHARE_FN',
                       'TRANS_SHARES_FN',
                       'TRANS_TIMELINESS_FN',
                       'EQUITY_SWAP_TRANS_CD_FN',
                       'TRANS_CODE',
                       'TRANS_FORM_TYPE',
                       'DEEMED_EXECUTION_DATE_FN',
                       'DEEMED_EXECUTION_DATE',
                       'TRANS_DATE_FN',
                       'SECURITY_TITLE_FN',
                       'SECURITY_TITLE']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    def correct_year_format(date_str):
        match = re.match(r'(\d{2}-\w{3}-00(\d{2}))', date_str)
        if match:
            corrected_year = date_str.replace('00', '20', 1)
            return corrected_year
        return date_str

    df['TRANS_DATE'] = df['TRANS_DATE'].apply(correct_year_format)

    df['EQUITY_SWAP_INVOLVED'] = df['EQUITY_SWAP_INVOLVED'].astype(str)
    df['EQUITY_SWAP_INVOLVED'] = df['EQUITY_SWAP_INVOLVED'].replace({
        'false': 'False',
        '0': 'False',
        '1': 'True',
        'true': 'True',
        'False': 'False',
        'True': 'True'
    })
    df['EQUITY_SWAP_INVOLVED'] = df['EQUITY_SWAP_INVOLVED'].map({'True': True, 'False': False})

    df['TRANS_TIMELINESS'] = df['TRANS_TIMELINESS'].replace(np.nan, 'O')
    df = df.dropna(subset=['SHRS_OWND_FOLWNG_TRANS', 'TRANS_PRICEPERSHARE'])

    return df

def process_submissions():
    files = [os.path.join('data', 'raw', 'insider_transactions', f'{year}q{quarter}_form345', 'SUBMISSION.tsv')
             for year in range(2014, 2018) for quarter in range(1, 5)]
    dataframes = []
    for file in files:
        if os.path.exists(file):
            try:
                temp = pd.read_csv(file, sep='\t', low_memory=False)
                dataframes.append(temp)
            except Exception as e:
                ...
                # print(f'Error reading {file}: {e}')
        else:
            print(f'File {file} does not exist')
    df2 = pd.concat(dataframes, ignore_index=True)
    columns_to_keep = ['ACCESSION_NUMBER', 'FILING_DATE', 'PERIOD_OF_REPORT', 'ISSUERNAME', 'ISSUERTRADINGSYMBOL']
    df2 = df2[columns_to_keep]
    issuer_symbol_map = df2.dropna(subset=['ISSUERTRADINGSYMBOL']).set_index('ISSUERNAME')['ISSUERTRADINGSYMBOL'].to_dict()
    df2['ISSUERTRADINGSYMBOL'] = df2.apply(
        lambda row: issuer_symbol_map.get(row['ISSUERNAME'], row['ISSUERTRADINGSYMBOL']) 
        if pd.isna(row['ISSUERTRADINGSYMBOL']) and pd.notna(row['ISSUERNAME']) else row['ISSUERTRADINGSYMBOL'],
        axis=1
    )
    df2.dropna(subset=['ISSUERTRADINGSYMBOL'], inplace=True)
    return df2

def process_reporting_owner():
    files = [os.path.join('data', 'raw', 'insider_transactions', f'{year}q{quarter}_form345', 'REPORTINGOWNER.tsv')
             for year in range(2014, 2018) for quarter in range(1, 5)]
    dataframes = []
    for file in files:
        if os.path.exists(file):
            try:
                temp = pd.read_csv(file, sep='\t', low_memory=False)
                dataframes.append(temp)
            except Exception as e:
                ...
                # print(f'Error reading {file}: {e}')
        else:
            print(f'File {file} does not exist')
    df3 = pd.concat(dataframes, ignore_index=True)
    columns_to_keep = ['RPTOWNER_RELATIONSHIP', 'ACCESSION_NUMBER']
    df3 = df3[columns_to_keep].dropna(subset=['RPTOWNER_RELATIONSHIP'])
    return df3

def process_stock_prices():
    files = [os.path.join('data', 'raw', 'stock_prices', filename) 
             for filename in os.listdir(os.path.join('data', 'raw', 'stock_prices')) 
             if filename.endswith('.txt')]
    dataframes = []
    for file in files:
        symbol = os.path.basename(file).split('.')[0]
        if os.path.exists(file):
            try:
                temp = pd.read_csv(file, sep=',', low_memory=False)
                temp['SYMBOL'] = symbol
                temp.drop(columns=['OpenInt'], inplace=True)
                temp = temp[temp['Date'].str.startswith(('2014', '2015', '2016', '2017'))]
                dataframes.append(temp)
            except Exception as e:
                print(f'{file}: {e}')
        else:
            print(f'File {file} does not exist')
    df5 = pd.concat(dataframes, ignore_index=True)
    return df5

def merge_data(df, df2, df3, df5):
    df4 = df.merge(df2, on='ACCESSION_NUMBER').merge(df3, on='ACCESSION_NUMBER')
    df4['ISSUERTRADINGSYMBOL'] = df4['ISSUERTRADINGSYMBOL'].str.upper()
    df5['SYMBOL'] = df5['SYMBOL'].str.upper()
    df4['TRANS_DATE'] = pd.to_datetime(df4['TRANS_DATE'], format='%d-%b-%Y').dt.strftime('%Y-%m-%d')
    merged_df = pd.merge(df4, df5, left_on=['ISSUERTRADINGSYMBOL', 'TRANS_DATE'], right_on=['SYMBOL', 'Date'], how='inner')
    return df4, df5, merged_df

def save_interim_data(df4, df5, merged_df):
    insider_transactions_path = os.path.join('data', 'interim', 'insider_transactions')
    stock_prices_path = os.path.join('data', 'interim', 'stock_prices')
    merged_path = os.path.join('data', 'interim', 'merged_insider_transactions_stock_prices')

    os.makedirs(insider_transactions_path, exist_ok=True)
    os.makedirs(stock_prices_path, exist_ok=True)
    os.makedirs(merged_path, exist_ok=True)

    df4.to_csv(os.path.join(insider_transactions_path, 'interim_insider_transactions.csv'), index=False)
    df5.to_csv(os.path.join(stock_prices_path, 'interim_stock_prices.csv'), index=False)
    merged_df.to_csv(os.path.join(merged_path, 'interim_merged_insider_transactions_stock_prices.csv'), index=False)

def extract_date_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date')
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df


def clean_data(df):
    df['RPTOWNER_RELATIONSHIP'] = df['RPTOWNER_RELATIONSHIP'].apply(
        lambda x: 'TenPercentOwner' if 'TenPercentOwner' in x else x
    )
    df['RPTOWNER_RELATIONSHIP'] = df['RPTOWNER_RELATIONSHIP'].apply(
        lambda x: 'Director' if 'Director' in x else x
    )
    df['RPTOWNER_RELATIONSHIP'] = df['RPTOWNER_RELATIONSHIP'].apply(
        lambda x: 'Officer' if 'Officer' in x else x
    )
    return df

def interim_further_processing(df_insider_transactions, df_stock_prices, df_merged):
        # Load and clean the merged dataframe
        df_merged = clean_data(df_merged)
        df_merged['TRANS_DATE'] = pd.to_datetime(df_merged['TRANS_DATE'])
        df_merged['TransactionValue'] = df_merged['TRANS_PRICEPERSHARE'] * df_merged['TRANS_SHARES']
        # Precompute a set of (Date, Symbol) tuples for faster lookup
        df_merged['Date_Symbol'] = df_merged['TRANS_DATE'].dt.strftime('%Y-%m-%d') + '_' + df_merged['SYMBOL']
        date_symbol_set = set(df_merged['Date_Symbol'])
        
        # Load and clean the stock prices dataframe
        df_stock_prices.loc[df_stock_prices['Low'] < 0, 'Low'] = df_stock_prices.loc[df_stock_prices['Low'] < 0, 'Open']
        df_stock_prices['Date'] = pd.to_datetime(df_stock_prices['Date'])
        df_stock_prices['DateNumeric'] = (df_stock_prices['Date'] - df_stock_prices['Date'].min()).dt.days
        df_stock_prices['MeanTotalValue'] = df_stock_prices['Volume'] * df_stock_prices[['Low', 'High', 'Open', 'Close']].mean(axis=1)
        # Vectorized check for 'Exists in Insiders'
        df_stock_prices['Date_Symbol'] = df_stock_prices['Date'].dt.strftime('%Y-%m-%d') + '_' + df_stock_prices['SYMBOL']
        df_stock_prices['Exists in Insiders'] = df_stock_prices['Date_Symbol'].isin(date_symbol_set)
        
        # Load and clean the insider transactions dataframe
        df_insider_transactions = clean_data(df_insider_transactions)
        df_insider_transactions['TRANS_DATE'] = pd.to_datetime(df_insider_transactions['TRANS_DATE'])
        df_insider_transactions['TransactionValue'] = df_insider_transactions['TRANS_PRICEPERSHARE'] * df_insider_transactions['TRANS_SHARES']
        # Precompute a set of (Date, Symbol) tuples for stock prices
        df_stock_prices['Date_Symbol'] = df_stock_prices['Date'].dt.strftime('%Y-%m-%d') + '_' + df_stock_prices['SYMBOL']
        date_symbol_set_stock_prices = set(df_stock_prices['Date_Symbol'])
        # Vectorized check for 'Exists in Stock Prices'
        df_insider_transactions['Date_Symbol'] = df_insider_transactions['TRANS_DATE'].dt.strftime('%Y-%m-%d') + '_' + df_insider_transactions['ISSUERTRADINGSYMBOL']
        df_insider_transactions['Exists in Stock Prices'] = df_insider_transactions['Date_Symbol'].isin(date_symbol_set_stock_prices)
        
        return df_insider_transactions, df_stock_prices, df_merged

def check_date_symbol(date, symbol):
    return (date, symbol) in date_symbol_set


def create_save_interim_data():
    print("Starting interim data processing...")
    total_steps = 3  # Define the number of steps for progress calculation
    current_step = 0

    # Define interim data paths
    insider_transactions_path = os.path.join(
        'data', 'interim', 'insider_transactions', 'interim_insider_transactions.csv'
    )
    stock_prices_path = os.path.join(
        'data', 'interim', 'stock_prices', 'interim_stock_prices.csv'
    )
    merged_path = os.path.join(
        'data', 'interim', 'merged_insider_transactions_stock_prices',
        'interim_merged_insider_transactions_stock_prices.csv'
    )

    # Check if interim data exists
    if (os.path.exists(insider_transactions_path) and
        os.path.exists(stock_prices_path) and
        os.path.exists(merged_path)):
        # Load interim data
        # df_insider_transactions = pd.read_csv(insider_transactions_path)
        # df_stock_prices = pd.read_csv(stock_prices_path)
        # df_merged = pd.read_csv(merged_path)
        print("CSV interim already exist.")
    else:
        # Process the data
        df = process_insider_transactions()
        df2 = process_submissions()
        df3 = process_reporting_owner()
        df_stock_prices = process_stock_prices()
        df_insider_transactions, df_stock_prices, df_merged = merge_data(df, df2, df3, df_stock_prices)
        current_step += 1
        print(f"Interim data processing: {(current_step / total_steps) * 100:.2f}% complete")
        df_insider_transactions, df_stock_prices, df_merged = interim_further_processing(df_insider_transactions, df_stock_prices, df_merged)
        current_step += 1
        print(f"Interim data processing: {(current_step / total_steps) * 100:.2f}% complete")
        save_interim_data(df_insider_transactions, df_stock_prices, df_merged)
        current_step += 1
        print(f"Interim data processing: {(current_step / total_steps) * 100:.2f}% complete")
        print("Interim saved successfully.")
    print("----------------------------\n")


def create_save_processed_data():
    print("Starting processed data saving...")
    total_steps = 3  # Define the number of steps for progress calculation
    current_step = 0

    # Define processed data paths
    insider_transactions_folder = os.path.join('data', 'processed', 'insider_transactions')
    stock_prices_folder = os.path.join('data', 'processed', 'stock_prices')
    merged_folder = os.path.join('data', 'processed', 'merged_insider_transactions_stock_prices')
    
    insider_transactions_file = os.path.join(insider_transactions_folder, 'processed_insider_transactions.csv')
    stock_prices_file = os.path.join(stock_prices_folder, 'processed_stock_prices.csv')
    merged_file = os.path.join(merged_folder, 'processed_merged_insider_transactions_stock_prices.csv')
    
    # Check if processed data exists
    if (os.path.exists(insider_transactions_file) and
        os.path.exists(stock_prices_file) and
        os.path.exists(merged_file)):
        print("CSV processed already exist")
    else:
        # Proceed with processing
        # ...existing code...

        # Load interim data
        insider_transactions_path = os.path.join(
            'data', 'interim', 'insider_transactions', 'interim_insider_transactions.csv'
        )
        stock_prices_path = os.path.join(
            'data', 'interim', 'stock_prices', 'interim_stock_prices.csv'
        )
        merged_path = os.path.join(
            'data', 'interim', 'merged_insider_transactions_stock_prices',
            'interim_merged_insider_transactions_stock_prices.csv'
        )

        df_insider_transactions = pd.read_csv(insider_transactions_path)
        df_stock_prices = pd.read_csv(stock_prices_path)
        df_merged = pd.read_csv(merged_path)
        current_step += 1
        print(f"Processed data saving: {(current_step / total_steps) * 100:.2f}% complete")
        # ...processing steps...
        current_step += 1
        print(f"Processed data saving: {(current_step / total_steps) * 100:.2f}% complete")
        # Save processed data
        os.makedirs(insider_transactions_folder, exist_ok=True)
        os.makedirs(stock_prices_folder, exist_ok=True)
        os.makedirs(merged_folder, exist_ok=True)
        df_insider_transactions.to_csv(insider_transactions_file, index=False)
        df_stock_prices.to_csv(stock_prices_file, index=False)
        df_merged.to_csv(merged_file, index=False)
        current_step += 1
        print(f"Processed data saving: {(current_step / total_steps) * 100:.2f}% complete")
        print("CSV processed saved successfully.")
    print("----------------------------\n")


def run_data_pipeline():
    print("Dataset ETL started.\n----------------------------")
    # get the directory of the current file and not the terminal directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    print("datapipeline script directory: ", current_dir)
    if current_dir.endswith('notebooks'):
        # go one directory up from the current directory
        os.chdir(os.path.dirname(current_dir))
        print("datapipeline script new current directory to: ", os.getcwd())
    print("----------------------------\n")
    # Define paths
    stock_prices_download_path = 'data/downloaded/zip_stock_prices/'
    insider_transactions_download_path = 'data/downloaded/zip_insider_transactions/'

    # create directories if they don't exist
    if not os.path.exists(stock_prices_download_path):
        os.makedirs(stock_prices_download_path)
    if not os.path.exists(insider_transactions_download_path):
        os.makedirs(insider_transactions_download_path)

    stock_prices_filename = "price-volume-data-for-all-us-stocks-etfs.zip"
    insider_transactions_filename = 'sec-insider-transactions.zip'

    stock_prices_unzip_path = 'data/raw/stock_prices/'
    insider_transactions_unzip_path = 'data/raw/insider_transactions/'


    zip_stock_prices_path = os.path.join(stock_prices_download_path, stock_prices_filename)
    zip_insider_transactions_path = os.path.join(insider_transactions_download_path, insider_transactions_filename)


    sec_insiders_url = 'https://suyvza.am.files.1drv.com/y4mFkHfFcjiHDfwnLRZYrYsI6LTW-Y8et8hQYjc-U1ePQ'+\
        '1bxs1YSBCUhkB7YVNpuz8wiadHeE7MBr3TjFQj2ELxVJV6ZI2X1_bvB797jFr6-7zU6g0rb9XJW0yqZTVGZv77LfyMyMV'+\
            'dw8M4P3mqZHTVpp4n7Z87NWPlOnKAc8leqq_ISYei2cow2uGGTwp1IgtbUfyNXRoWxl8RBeH8W2_kgQ'
    #'https://1drv.ms/u/s!AizgSozl2n1gh9NtvBQEuLK4Yq_rTg?e=wfhw2x'
    stocks_url = 'https://g0qmta.am.files.1drv.com/y4man0AxtAeofAVOne5Mv07S5foqCe3cteOTeSIBIRyW_P4f0HdvK'+\
        'FukP02zqzRJV-2tPXehTzoVE_GORGSZDi9h06HQgwIv8QrNppjvjTUo_VEs4DIEILahLm4qc6SVN_edgFp6VkaCY-wfDGsRm'+\
            'Mz5x3_CJKu5UAAut6jkfjL5ChzWNUnP6Tuq3CIVikFC2-FAny3KAPzWZDgIaBwptN-BQ'
    # 'https://1drv.ms/u/s!AizgSozl2n1gh9NuhIhU6GCuMCTZmQ?e=z1Iro6'


    # Download the insider transactions file
    download_zip_file(sec_insiders_url, insider_transactions_download_path, insider_transactions_filename)

    # Download the stock prices file
    download_zip_file(stocks_url, stock_prices_download_path, stock_prices_filename)

    # Unzip files
    unzip_stock_prices(zip_stock_prices_path, stock_prices_unzip_path)
    unzip_insider_transactions(zip_insider_transactions_path, insider_transactions_unzip_path)

    create_save_interim_data()
    create_save_processed_data()
    print("Dataset ETL successful.\n----------------------------")

if __name__ == "__main__":
    run_data_pipeline()


