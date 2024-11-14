# app.py
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import os
from dotenv import load_dotenv
import zipfile
import shutil

# Data preparation imports
import subprocess
import pandas as pd
import numpy as np

# Importing layouts from other files
from data_exploration_app import layout as exploration_layout
from final_app_prediction import layout as prediction_layout

# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose the server for deployment

# Load environment variables from the .env file if it exists
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(".env file loaded from:", env_path)
else:
    print("No .env file found. Ensure environment variables are set in the hosting environment.")

# Access environment variables
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

# Verify environment variables are set
if not kaggle_username or not kaggle_key:
    print("Warning: KAGGLE_USERNAME and/or KAGGLE_KEY environment variables are not set.")
else:
    print("Environment variables loaded successfully.")

# Define paths for downloading and unzipping datasets
stock_prices_download_path = 'data/downloaded/zip_stock_prices/'
stock_prices_filename = "price-volume-data-for-all-us-stocks-etfs.zip"
stock_prices_unzip_path = 'data/raw/stock_prices/'

insider_transactions_download_path = 'data/downloaded/zip_insider_transactions/'
insider_transactions_filename = 'sec-insider-transactions.zip'
insider_transactions_unzip_path = 'data/raw/insider_transactions/'

# Define dataset names for Kaggle CLI
stock_prices_dataset = "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
insider_transactions_dataset = "osawani/sec-insider-transactions"

# Function to download the dataset using Kaggle CLI
def download_dataset(dataset_name, download_path):
    command = f"kaggle datasets download -d {dataset_name} -p {download_path}"
    print(f"Running command: {command}")
    os.system(command)
    print(f"Dataset {dataset_name} downloaded successfully to {download_path}")

# Function to check if the file exists in the folder and download it if it doesn't
def check_and_download_file(folder_path, filename, dataset_name):
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        print(f"File {filename} already exists in {folder_path}")
        return True  # File exists
    else:
        print(f"File {filename} does not exist in {folder_path}. Downloading now...")
        download_dataset(dataset_name, folder_path)
        return False  # File does not exist, download initiated

# Function to unzip the Stock Prices dataset (only the 'Stocks' folder)
def unzip_stock_prices(zip_path, unzip_path, specific_folder='Stocks'):
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist. Please download the zip file first.")
        return

    zip_dir = os.path.dirname(zip_path)
    print(f"Unzipping {zip_path} to {zip_dir}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        files_to_extract = [file for file in all_files if file.startswith(specific_folder)]
        
        for file in files_to_extract:
            zip_ref.extract(file, zip_dir)

    stocks_folder_path = os.path.join(zip_dir, specific_folder)
    if os.path.exists(stocks_folder_path):
        for item in os.listdir(stocks_folder_path):
            source = os.path.join(stocks_folder_path, item)
            destination = os.path.join(unzip_path, item)
            if os.path.isdir(source):
                shutil.move(source, destination)
            else:
                shutil.move(source, destination)
        
        shutil.rmtree(stocks_folder_path)
        print(f"Moved contents from '{stocks_folder_path}' to '{unzip_path}' and deleted the folder.")
    else:
        print(f"Error: The folder '{specific_folder}' was not found in the zip file.")

    print(f"Unzipping complete. Files extracted and moved to {unzip_path}.")

# Function to unzip the Insider Transactions dataset (extract everything)
def unzip_insider_transactions(zip_path, unzip_path):
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist. Please download the zip file first.")
        return

    print(f"Unzipping {zip_path} to {unzip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)

    print(f"Unzipping complete. Files extracted to {unzip_path}.")

# Main process to check and download datasets, then unzip them
def prepare_data():
    # 1. Check and download the Stock Prices dataset if the file doesn't exist
    check_and_download_file(stock_prices_download_path, stock_prices_filename, stock_prices_dataset)

    # 2. Check and download the Insider Transactions dataset if the file doesn't exist
    check_and_download_file(insider_transactions_download_path, insider_transactions_filename, insider_transactions_dataset)

    # 3. Unzip the Stock Prices dataset (only the 'Stocks' folder)
    zip_stock_prices_path = os.path.join(stock_prices_download_path, stock_prices_filename)
    unzip_stock_prices(zip_stock_prices_path, stock_prices_unzip_path)

    # 4. Unzip the Insider Transactions dataset (all contents)
    zip_insider_transactions_path = os.path.join(insider_transactions_download_path, insider_transactions_filename)
    unzip_insider_transactions(zip_insider_transactions_path, insider_transactions_unzip_path)

    print("Download and extraction complete!")

# Call the data preparation function when the app is initialized
prepare_data()

# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dcc.Link("Data Exploration", href="/data_exploration", className="nav-link")),
        dbc.NavItem(dcc.Link("Predictions", href="/predictions", className="nav-link")),
    ],
    brand="Stocks & Insiders App",
    color="primary",
    dark=True,
)

# Define the content placeholder
content = html.Div(id="page-content")

# Set up the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    content
])

# Update page content based on the URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/' or pathname == '/data_exploration':
        return exploration_layout
    elif pathname == '/predictions':
        return prediction_layout
    else:
        return html.H1("404: Page not found", className='text-center')

# Run the app server
if __name__ == '__main__':
    app.run_server(debug=False)  # Disabled debug mode for production