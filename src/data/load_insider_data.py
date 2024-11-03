# 1. Import the necessary libraries and set up the file paths.
import pandas as pd
import os
import glob

# Define the path to the raw data
raw_data_path = 'data/raw/insider_transactions/'

# Define the quarters and years you want to process
years = range(2014, 2019)  # Example: 2014 to 2018
quarters = ['QTR1', 'QTR2', 'QTR3', 'QTR4']



# 2. Function to Load TSV File
def load_insider_tsv_files(data_path, years, quarters, table_name):
    df_list = []
    for year in years:
        for quarter in quarters:
            folder_path = os.path.join(data_path, f'{year}', f'{quarter}')
            file_path = os.path.join(folder_path, f'{table_name}.tsv')
            if os.path.exists(file_path):
                temp_df = pd.read_csv(file_path, sep='\t', encoding='utf-8', low_memory=False)
                temp_df['Year'] = year
                temp_df['Quarter'] = quarter
                df_list.append(temp_df)
            else:
                print(f"File not found: {file_path}")
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df



# 3. Load each table
# Load SUBMISSION table
submission_df = load_insider_tsv_files(raw_data_path, years, quarters, 'SUBMISSION')

# Load REPORTINGOWNER table
reportingowner_df = load_insider_tsv_files(raw_data_path, years, quarters, 'REPORTINGOWNER')

# Load NONDERIV_TRANS table
nonderiv_trans_df = load_insider_tsv_files(raw_data_path, years, quarters, 'NONDERIV_TRANS')



# 4. Merge DataFrames using the ACCESSION_NUMBER as the primary key
# Merge SUBMISSION and NONDERIV_TRANS on ACCESSION_NUMBER
merged_df = pd.merge(nonderiv_trans_df, submission_df, on='ACCESSION_NUMBER', how='inner')

# Merge with REPORTINGOWNER on ACCESSION_NUMBER
merged_df = pd.merge(merged_df, reportingowner_df, on='ACCESSION_NUMBER', how='inner')


# 5. Data Cleaning: Identify columns with missing values in order to decide on a strategy to handle them
# Check for missing values
missing_values = merged_df.isnull().sum()
print(missing_values)

# For critical fields like TRANS_DATE, ISSUERTRADINGSYMBOL, or TRANS_SHARES, we may choose to drop rows with missing values.
# Drop rows with missing critical values
critical_columns = ['TRANS_DATE', 'ISSUERTRADINGSYMBOL', 'TRANS_SHARES']
merged_df.dropna(subset=critical_columns, inplace=True)



# Standardize Date Formats: Convert date fields to datetime objects.
# Convert TRANS_DATE to datetime
merged_df['TRANS_DATE'] = pd.to_datetime(merged_df['TRANS_DATE'], format='%d-%b-%Y', errors='coerce')

# Handle conversion errors
merged_df.dropna(subset=['TRANS_DATE'], inplace=True)


# Decode Transaction Codes: use the transaction code mappings provided in the SEC documentation to replace codes with meaningful descriptions.
# Define transaction code mappings
transaction_code_mapping = {
    'P': 'Purchase',
    'S': 'Sale',
    # Add other codes as needed
}

# Map TRANS_CODE to descriptions
merged_df['TRANS_CODE_DESC'] = merged_df['TRANS_CODE'].map(transaction_code_mapping)



# Filter Data: If needed, we will filter data for a specific date range or companies.
# Filter data for transactions after a certain date
start_date = '2014-01-01'
merged_df = merged_df[merged_df['TRANS_DATE'] >= start_date]
# Rename Columns for Clarity: Rename columns to more readable names.
merged_df.rename(columns={
    'ISSUERTRADINGSYMBOL': 'Ticker',
    'ISSUERNAME': 'Company Name',
    'RPTOWNERNAME': 'Insider Name',
    'RPTOWNER_TITLE': 'Insider Title',
    'TRANS_DATE': 'Trade Date',
    'TRANS_SHARES': 'Shares Traded',
    'TRANS_PRICEPERSHARE': 'Price per Share',
    'SHRS_OWND_FOLWNG_TRANS': 'Owned Shares After Transaction',
    'VALU_OWND_FOLWNG_TRANS': 'Value Owned After Transaction'
}, inplace=True)



# Final DataFrame Preview
print(merged_df.head())
