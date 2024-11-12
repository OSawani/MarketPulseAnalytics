# app.py
from dash import Dash
import dash_bootstrap_components as dbc
import os
from dotenv import load_dotenv
import zipfile
import shutil


# Change working directory
current_dir = os.getcwd()
current_dir
os.chdir(os.path.dirname(current_dir))
# print("You set a new current directory")



# Kaggle credintials will be read from Heroku in production



# Set Kaggle directory
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()




# Download the dataset: setting download paths
KaggleDatasetPath = "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
DestinationFolder = "data/downloaded/stock_prices"
# Make sure the destination folder exists
os.makedirs(DestinationFolder, exist_ok=True)
# Check if the dataset zip file already exists
if not os.path.exists():
    # If the zip file doesn't exist, download it using Kaggle API
    print(f"Dataset not found. Downloading {KaggleDatasetPath}...")
    os.system(f"kaggle datasets download -d {KaggleDatasetPath} -p {DestinationFolder}")
else:
    # If the zip file exists, skip the download
    print(f"Dataset already exists at {DestinationFolder, "price-volume-data-for-all-us-stocks-etfs.zip"}. Skipping download.")



# Unzip the downloaded dataset
# Path to the downloaded ZIP file
zip_file_path = os.path.join(DestinationFolder, "price-volume-data-for-all-us-stocks-etfs.zip")

# Unzip the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(DestinationFolder)
# Define paths for ETFs folder, Stocks folder, and target folder
etfs_folder = os.path.join(DestinationFolder, "ETFs")
stocks_folder = os.path.join(DestinationFolder, "Stocks")
target_folder = os.path.join("data", "raw", "stock_prices")
# Delete the ETFs folder if it exists
if os.path.exists(etfs_folder):
    shutil.rmtree(etfs_folder)
# Move the content of the "Stocks" folder to the target "stock_prices" folder
if os.path.exists(stocks_folder):
    for file_name in os.listdir(stocks_folder):
        src_file = os.path.join(stocks_folder, file_name)
        dest_file = os.path.join(target_folder, file_name)
        
        # Check if the destination file already exists
        if os.path.exists(dest_file):
            print(f"File {file_name} already exists in the target directory. Skipping.")
        else:
            shutil.move(src_file, dest_file)

    # Remove the now-empty "Stocks" folder
    shutil.rmtree(stocks_folder)
# Delete the original ZIP file if it exists
if os.path.exists(zip_file_path):
    os.remove(zip_file_path)



# Unzipping SEC Dataset
# Define the project root by finding "MarketPulseAnalytics" within the directory structure
current_dir = os.getcwd()
# Navigate up to the "Repo" level, where "MarketPulseAnalytics" resides
project_root = os.path.join(current_dir, "..", "MarketPulseAnalytics")
# Change to the project root directory
os.chdir(project_root)
print("New current directory:", os.getcwd())
# Define paths relative to the project root
downloaded_folder = os.path.join("data", "downloaded", "zipped_insider_transactions")
extracted_folder = os.path.join("data", "raw", "insider_transactions")
# Ensure both the extracted folders exist
os.makedirs(extracted_folder, exist_ok=True)
# Check and list all ZIP files in the downloaded folder
if not os.path.exists(downloaded_folder):
    print(f"No folder named 'downloaded' found in {downloaded_folder}.")
else:
    zip_files = [f for f in os.listdir(downloaded_folder) if f.endswith('.zip')]
    if not zip_files:
        print("No ZIP files found in the downloaded folder.")
    else:
        # Process each ZIP file
        for zip_file in zip_files:
            zip_file_path = os.path.join(downloaded_folder, zip_file)
            
            # Create a new folder named after the ZIP file (without .zip) for extraction
            zip_folder_name = os.path.splitext(zip_file)[0]
            destination_folder = os.path.join(extracted_folder, zip_folder_name)
            os.makedirs(destination_folder, exist_ok=True)
            print(f"Extracting {zip_file} into {destination_folder}...")

            # Extract the contents of the ZIP file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)

        print("All ZIP files have been extracted into their respective folders.")


app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose the server for deployment
