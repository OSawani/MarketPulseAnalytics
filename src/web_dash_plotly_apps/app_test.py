# app.py
import os
import sys
from dotenv import load_dotenv
import zipfile
import dash
from dash import dcc, html, Dash
import dash_bootstrap_components as dbc

# Add the src directory to the system path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from web_dash_plotly_apps.data_exploration_app import app as app1  # Absolute import
from web_dash_plotly_apps.final_app_prediction import app as app2  # Absolute import

# # Load environment variables from the .env file if it exists
# env_path = os.path.join(os.getcwd(), '.env')
# if os.path.exists(env_path):
#     load_dotenv(env_path)
#     print(".env file loaded from:", env_path)
# else:
#     print("No .env file found. Ensure environment variables are set in the hosting environment.")

# # Access environment variables
# kaggle_username = os.getenv('KAGGLE_USERNAME')
# kaggle_key = os.getenv('KAGGLE_KEY')

# # Verify environment variables are set
# if not kaggle_username or not kaggle_key:
#     print("Warning: KAGGLE_USERNAME and/or KAGGLE_KEY environment variables are not set.")
# else:
#     print("Environment variables loaded successfully.")

# # Define paths for downloading and unzipping datasets
# data_download_path = 'data/downloaded/interim_and_processed/'
# data_filename = "interim-and-processed-datasets"
# data_unzip_path = 'data/'

# # Define dataset names for Kaggle CLI
# dataset = "osawani/interim-and-processed-datasets"

# # Function to download the dataset using Kaggle CLI
# def download_dataset(dataset_name, download_path):
#     command = f"kaggle datasets download -d {dataset_name} -p {download_path}"
#     print(f"Running command: {command}")
#     os.system(command)
#     print(f"Dataset {dataset_name} downloaded successfully to {download_path}")

# # Function to check if the file exists in the folder and download it if it doesn't
# def check_and_download_file(folder_path, filename, dataset_name):
#     file_path = os.path.join(folder_path, filename)
#     if os.path.exists(file_path):
#         print(f"File {filename} already exists in {folder_path}")
#         return True  # File exists
#     else:
#         print(f"File {filename} does not exist in {folder_path}. Downloading now...")
#         download_dataset(dataset_name, folder_path)
#         return False  # File does not exist, download initiated

# # Function to unzip the dataset (extract everything)
# def unzip_data(zip_path, unzip_path):
#     if not os.path.exists(zip_path):
#         print(f"Error: {zip_path} does not exist. Please download the zip file first.")
#         return

#     print(f"Unzipping {zip_path} to {unzip_path}...")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(unzip_path)

#     print(f"Unzipping complete. Files extracted to {unzip_path}.")

# # Main process to check and download datasets, then unzip them
# def prepare_data():
#     # 1. Check and download the Stock Prices dataset if the file doesn't exist
#     check_and_download_file(data_download_path, data_filename, dataset)

#     # 2. Unzip the Stock Prices dataset (only the 'Stocks' folder)
#     zip_data = os.path.join(data_download_path, data_filename)
#     unzip_data(zip_data, data_unzip_path)

#     print("Download and extraction complete!")

# # Call the data preparation function when the app is initialized
# prepare_data()

# Initialize the landing page app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the landing page
app.layout = html.Div(
    children=[
        # Project Title and Summary Section
        html.H1("Stock Predictions and Data Exploration Project", style={"text-align": "center", "margin-top": "20px"}),
        
        html.Div(
            children=[
                html.P("Welcome to the Stocks & Insiders Predictions project. This platform allows users to:"),
                html.Ul(
                    children=[
                        html.Li("Explore stock data and trends."),
                        html.Li("Make stock price predictions based on historical data."),
                        html.Li("Analyze insider trading activities."),
                    ],
                    style={"text-align": "left", "margin-left": "20px"}
                ),
            ],
            style={"margin": "20px"}
        ),

        # Links to the Prediction and Exploration Apps
        html.Div(
            children=[
                html.H3("Explore the Apps", style={"text-align": "center", "margin-top": "40px"}),

                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                dbc.Card(
                                    children=[
                                        dbc.CardHeader("Prediction App"),
                                        dbc.CardBody(
                                            html.P("Predict stock prices based on historical data and trends."),
                                            dbc.Button("Go to Prediction App", href="http://127.0.0.1:32335", color="primary"),
                                        ),
                                    ],
                                ),
                            ],
                            width=5,
                            style={"margin": "10px"}
                        ),
                        dbc.Col(
                            children=[
                                dbc.Card(
                                    children=[
                                        dbc.CardHeader("Data Exploration App"),
                                        dbc.CardBody(
                                            html.P("Explore stock data and analyze trends."),
                                            dbc.Button("Go to Data Exploration App", href="http://127.0.0.1:32336", color="primary"),
                                        ),
                                    ],
                                ),
                            ],
                            width=5,
                            style={"margin": "10px"}
                        ),
                    ],
                    justify="center",
                ),
            ],
            style={"margin": "20px"}
        ),

        # Footer Section
        html.Div(
            children=[
                html.P("Developed by [Your Name].", style={"text-align": "center", "font-size": "14px", "margin-top": "50px"}),
            ],
            style={"position": "fixed", "bottom": "10px", "width": "100%", "text-align": "center"}
        ),
    ],
)

# Get the port from the environment variable
port = os.environ.get("PORT", 32334)

# Run the landing page app
if __name__ == "__main__":
    app.run_server(debug=True, port=port, host="0.0.0.0")