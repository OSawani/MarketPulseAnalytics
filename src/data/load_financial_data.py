# Load the financial indicators from the Kaggle dataset for years 2014-2018 and combine them into a single DataFrame

# 1. Load CSV Files
financial_data_path = 'data/raw/financial_indicators/'

# List of financial data files
financial_files = [
    '2014_Financial_Data.csv',
    '2015_Financial_Data.csv',
    '2016_Financial_Data.csv',
    '2017_Financial_Data.csv',
    '2018_Financial_Data.csv'
]

# Load and combine the financial data
financial_dfs = []
for file in financial_files:
    file_path = os.path.join(financial_data_path, file)
    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    # Extract the year from the file name
    df['Fiscal Year'] = int(file.split('_')[0])
    financial_dfs.append(df)

financial_df = pd.concat(financial_dfs, ignore_index=True)




# 2. Standardize Column Names: Ensure that the columns have consistent and meaningful names.
# Rename columns
financial_df.rename(columns={'Unnamed: 0': 'Ticker'}, inplace=True)



# 3. Data Cleaning: Handle Missing Values
# Check for missing values
missing_values = financial_df.isnull().sum()
print(missing_values)

# Decide on imputation or removal
threshold = 0.5  # Example: remove columns with more than 50% missing values
financial_df = financial_df.loc[:, financial_df.isnull().mean() < threshold]

# Fill remaining missing values with median or mean
financial_df.fillna(financial_df.median(), inplace=True)



# 4. Handle Outliers: Identify and handle outliers in the financial indicators.
# Use Z-score to identify outliers
from scipy import stats
import numpy as np

numeric_cols = financial_df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate Z-scores
z_scores = np.abs(stats.zscore(financial_df[numeric_cols]))
outliers = (z_scores > 3)

# Replace outliers with median values
for col in numeric_cols:
    median_value = financial_df[col].median()
    financial_df.loc[outliers[col], col] = median_value
