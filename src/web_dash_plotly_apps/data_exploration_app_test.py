import pandas as pd
import os
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import plotly.io as pio
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error# deprecated
from sklearn.metrics import root_mean_squared_error# alternative

from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

# Initialize the Dash app for Data Exploration
app = Dash('Stocks & Insiders App', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the folder paths
# get the path of the current file without the file name
path_current_file_without_file_name = os.getcwd()
insider_transactions_path = os.path.join(path_current_file_without_file_name, 'data', 'interim', 'insider_transactions','interim_insider_transactions.csv')
stock_prices_path = os.path.join(path_current_file_without_file_name,'data', 'interim', 'stock_prices','interim_stock_prices.csv')
merged_path = os.path.join(path_current_file_without_file_name,'data', 'interim', 'merged_insider_transactions_stock_prices','interim_merged_insider_transactions_stock_prices.csv')
df_insider_transactions = pd.read_csv(insider_transactions_path)
df_stock_prices = pd.read_csv(stock_prices_path)
df_merged = pd.read_csv(merged_path)
# --------------------------------------------


def extract_date_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date')
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    # return df.drop(columns='Date')
    return df
def clean_data(df):
    df['RPTOWNER_RELATIONSHIP'] = df['RPTOWNER_RELATIONSHIP'].apply(lambda x: 'TenPercentOwner' if 'TenPercentOwner' in x else x)
    df['RPTOWNER_RELATIONSHIP'] = df['RPTOWNER_RELATIONSHIP'].apply(lambda x: 'Director' if 'Director' in x else x)
    df['RPTOWNER_RELATIONSHIP'] = df['RPTOWNER_RELATIONSHIP'].apply(lambda x: 'Officer' if 'Officer' in x else x)
    return df
def check_date_symbol(date, symbol):
    return (date, symbol) in date_symbol_set

# ------------------------------------------------------------------------------
# Load and clean the merged dataframe
df_merged = pd.read_csv(merged_path)
df_merged = clean_data(df_merged)
df_merged['TRANS_DATE'] = pd.to_datetime(df_merged['TRANS_DATE'])
df_merged['TransactionValue'] = df_merged['TRANS_PRICEPERSHARE'] * df_merged['TRANS_SHARES']
# Precompute a set of (Date, Symbol) tuples for faster lookup
df_merged['Date_Symbol'] = df_merged['TRANS_DATE'].dt.strftime('%Y-%m-%d') + '_' + df_merged['SYMBOL']
date_symbol_set = set(df_merged['Date_Symbol'])
# ------------------------------------------------------------------------------
# Load and clean the stock prices dataframe
df_stock_prices = pd.read_csv(stock_prices_path)
df_stock_prices.loc[df_stock_prices['Low'] < 0, 'Low'] = df_stock_prices.loc[df_stock_prices['Low'] < 0, 'Open']
df_stock_prices['Date'] = pd.to_datetime(df_stock_prices['Date'])
df_stock_prices['DateNumeric'] = (df_stock_prices['Date'] - df_stock_prices['Date'].min()).dt.days
df_stock_prices['MeanTotalValue'] = df_stock_prices['Volume'] * df_stock_prices[['Low', 'High', 'Open', 'Close']].mean(axis=1)
# Vectorized check for 'Exists in Insiders'
df_stock_prices['Date_Symbol'] = df_stock_prices['Date'].dt.strftime('%Y-%m-%d') + '_' + df_stock_prices['SYMBOL']
df_stock_prices['Exists in Insiders'] = df_stock_prices['Date_Symbol'].isin(date_symbol_set)
# ------------------------------------------------------------------------------
# Load and clean the insider transactions dataframe
df_insider_transactions = pd.read_csv(insider_transactions_path)
df_insider_transactions = clean_data(df_insider_transactions)
df_insider_transactions['TRANS_DATE'] = pd.to_datetime(df_insider_transactions['TRANS_DATE'])
df_insider_transactions['TransactionValue'] = df_insider_transactions['TRANS_PRICEPERSHARE'] * df_insider_transactions['TRANS_SHARES']
# Precompute a set of (Date, Symbol) tuples for stock prices
df_stock_prices['Date_Symbol'] = df_stock_prices['Date'].dt.strftime('%Y-%m-%d') + '_' + df_stock_prices['SYMBOL']
date_symbol_set_stock_prices = set(df_stock_prices['Date_Symbol'])
# Vectorized check for 'Exists in Stock Prices'
df_insider_transactions['Date_Symbol'] = df_insider_transactions['TRANS_DATE'].dt.strftime('%Y-%m-%d') + '_' + df_insider_transactions['ISSUERTRADINGSYMBOL']
df_insider_transactions['Exists in Stock Prices'] = df_insider_transactions['Date_Symbol'].isin(date_symbol_set_stock_prices)



def launch_stocks_insiders_exploration_app():        
    # Initialize the Dash app with Bootstrap
    app = Dash('Stocks & Insiders App', external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Define the app layout
    app.layout = html.Div([
        dbc.Container([
            html.H1("Stocks & Insiders Activities", className='text-center mb-4'),
            dbc.Row([
                # Left Column: Stocks Layout
                dbc.Col([
                    html.H4("Stock Prices Controls"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Choose a symbol:"),
                            dcc.Dropdown(
                                id='symbol',
                                options=[{'label': i, 'value': i} for i in df_stock_prices['SYMBOL'].unique()],
                                value='AAPL',
                                clearable=False,
                                style={'backgroundColor': '#ffffff', 'color': 'white'}  
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Choose a column:"),
                            dcc.Dropdown(
                                id='column',
                                options=[
                                    {'label': 'Low', 'value': 'Low'},
                                    {'label': 'High', 'value': 'High'},
                                    {'label': 'Close', 'value': 'Close'},
                                    {'label': 'Open', 'value': 'Open'}
                                ],
                                value='Low',
                                clearable=False,
                                style={'backgroundColor': '#ffffff', 'color': 'white'} 
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Choose a date range:"),
                        ], width=2),
                        dbc.Col([
                            dcc.DatePickerRange(
                                id='date_range',
                                start_date=df_stock_prices[df_stock_prices['Date'].dt.year == 2014]['Date'].min(),
                                end_date=df_stock_prices[df_stock_prices['Date'].dt.year == 2014]['Date'].max(),
                                style={'backgroundColor': '#ffffff', 'color': 'white'}  
                            ),
                        ], width=4),
                        dbc.Col([
                            dcc.Checklist(
                                id='theme-toggle',
                                options=[
                                    {'label': 'Dark Mode', 'value': 'dark'}
                                ],
                                value=[]
                            ),
                        ], width=3),
                        dbc.Col([
                            dcc.RadioItems(
                                id='display_mode',
                                options=[
                                    {'label': 'Scatter Points', 'value': 'scatter'},
                                    {'label': 'Lines', 'value': 'lines'},
                                    {'label': 'Both', 'value': 'both'}
                                ],
                                value='both',
                                inline=True
                            )
                        ], width=3)
                    ]),
                    dcc.Graph(id='stock_prices', config={'responsive': True})
                ], width=12, lg=6),
                # Right Column: Insiders Layout
                dbc.Col([
                    html.H4("Insiders Trading Controls"),
                    html.Label("Choose a symbol:"),
                    dcc.Dropdown(
                        id='symbol2',
                        options=[{'label': i, 'value': i} for i in df_merged['ISSUERTRADINGSYMBOL'].unique()],
                        value='AAPL',
                        clearable=False,
                        style={'backgroundColor': '#ffffff', 'color': 'white'}  # Initial dark mode style
                    ),
                    html.Label("Choose a date range:"),
                    dcc.DatePickerRange(
                        id='date_range2',
                        start_date=df_merged[df_merged['TRANS_DATE'].dt.year == 2014]['TRANS_DATE'].min(),
                        end_date=df_merged[df_merged['TRANS_DATE'].dt.year == 2014]['TRANS_DATE'].max(),
                        style={'backgroundColor': '#ffffff', 'color': 'white'}  # Initial dark mode style
                    ),
                    dcc.Graph(id='insiders_trading', config={'responsive': True})
                ], width=12, lg=6)
            ])
        ], fluid=True)
    ], id='main-div', style={'backgroundColor': '#f8f9fa'})  # Light mode default

    # Define a function to style components based on theme
    def get_component_style(theme):
        if False:# or 'dark' in theme:
            return {
                'backgroundColor': '#2c2c2c',  # Dark background for dropdowns and date pickers
                'color': 'red',
                'border': '1px solid #444444',
            }
        else:
            return {
                'backgroundColor': '#E7E0E0',  # Light background for dropdowns and date pickers
                'color': 'black',
                'border': '1px solid #cccccc',
            }


    @app.callback(
        Output('stock_prices', 'figure'),
        Output('insiders_trading', 'figure'),
        Output('main-div', 'style'),
        Output('symbol', 'style'),  
        Output('column', 'style'),  
        Output('date_range', 'style'),  
        Output('symbol2', 'style'),  
        Output('date_range2', 'style'),  
        [Input('symbol', 'value'),
        Input('column', 'value'),
        Input('symbol2', 'value'),
        Input('date_range2', 'start_date'),
        Input('date_range2', 'end_date'),
        Input('date_range', 'start_date'),
        Input('date_range', 'end_date'),
        Input('theme-toggle', 'value'),
        Input('display_mode', 'value')]
    )


    def update_figure(symbol, column, symbol2, start_date2, end_date2, start_date1, end_date1, theme, display_mode):
        # Prepare data for stock prices
        df = df_stock_prices[(df_stock_prices['SYMBOL'] == symbol) &
                            (df_stock_prices['Date'] >= start_date1) &
                            (df_stock_prices['Date'] <= end_date1)].copy()
        # Prepare data for insiders trading
        df2 = df_insider_transactions[(df_insider_transactions['ISSUERTRADINGSYMBOL'] == symbol2) &
                                    (df_insider_transactions['TRANS_DATE'] >= start_date2) &
                                    (df_insider_transactions['TRANS_DATE'] <= end_date2)].copy()
        
        dropdown_style = get_component_style(theme)

        # Determine theme styles
        if 'dark' in theme:
            main_div_style = {'backgroundColor': '#2c2c2c', 'color': 'white'}
        else:
            main_div_style = {'backgroundColor': '#f8f9fa', 'color': 'black'}

        # Get styles for each component
        symbol_style = dropdown_style
        column_style = dropdown_style
        date_range_style = dropdown_style
        symbol2_style = dropdown_style
        date_range2_style = dropdown_style

        # Create figure for stock prices
        fig_stock_prices = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=[
                f"{column} vs Date (size by Volume)",
                f"{column} vs Date (size by MeanTotalValue)",
                "Price Lines Over Time"
            ]
        )

        if display_mode in ['scatter', 'both']:
            scatter_volume = px.scatter(
                df,
                x='Date',
                y=column,
                size='Volume',
                color='DateNumeric',
                color_continuous_scale='Viridis',
                hover_data={
                    'Date': '|%Y-%m-%d',
                    column: ':.2s',
                    'Volume': ':.2s'
                }
            )
            scatter_volume.update_coloraxes(colorbar=dict(title='Recency', tickvals=[]))
            for trace in scatter_volume.data:
                fig_stock_prices.add_trace(trace, row=1, col=1)

            scatter_mean_total = px.scatter(
                df,
                x='Date',
                y=column,
                size='MeanTotalValue',
                color='DateNumeric',
                color_continuous_scale='Viridis',
                hover_data={
                    'Date': '|%Y-%m-%d',
                    column: ':.2s',
                    'MeanTotalValue': ':.2s'
                }
            )
            scatter_mean_total.update_coloraxes(colorbar=dict(title='Recency', tickvals=[]))
            for trace in scatter_mean_total.data:
                fig_stock_prices.add_trace(trace, row=2, col=1)

        if display_mode in ['lines', 'both']:
            for line_name, line_color in zip(['Open', 'High', 'Low', 'Close'], ['gray', 'magenta', 'darkblue', 'green']):
                fig_stock_prices.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df[line_name],
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        name=line_name,
                        hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>{line_name}: %{{y:.2s}}<extra></extra>'
                    ),
                    row=3, col=1
                )

        fig_stock_prices.update_layout(
            plot_bgcolor=main_div_style['backgroundColor'],
            paper_bgcolor=main_div_style['backgroundColor'],
            font=dict(color=main_div_style['color']),
            height=800,
            title_text='Stock Prices',
            legend=dict(
                orientation='h',
                x=0.5,
                y=1.05,
                xanchor='center',
                yanchor='bottom',
                traceorder='normal',
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(title='', showticklabels=True),
            autosize=True
        )

        # Create figure for insiders trading
        fig_insiders_trading = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=[
                "TransactionValue vs TRANS_DATE (size by SHRS_OWND_FOLWNG_TRANS)",
                "DIRECT_INDIRECT_OWNERSHIP vs TRANS_DATE (size by TransactionValue)",
                "TransactionValue vs TRANS_DATE"
            ]
        )

        scatter_owner_shares = px.scatter(
            df2,
            x='TRANS_DATE',
            y='TransactionValue',
            size='SHRS_OWND_FOLWNG_TRANS',
            color='RPTOWNER_RELATIONSHIP',
            hover_data={
                'TRANS_DATE': '|%Y-%m-%d',
                'TransactionValue': ':.2s',
                'SHRS_OWND_FOLWNG_TRANS': ':.2s'
            }
        )
        for trace in scatter_owner_shares.data:
            fig_insiders_trading.add_trace(trace, row=1, col=1)

        scatter_transaction_amount = px.scatter(
            df2,
            x='TRANS_DATE',
            y='DIRECT_INDIRECT_OWNERSHIP',
            size='TransactionValue',
            color='TRANS_ACQUIRED_DISP_CD',
            color_discrete_map={'A': 'green', 'D': 'yellow'},
            hover_data={
                'TRANS_DATE': '|%Y-%m-%d',
                'TransactionValue': ':.2s'
            }
        )
        for trace in scatter_transaction_amount.data:
            fig_insiders_trading.add_trace(trace, row=2, col=1)

        line_fig = px.scatter(
            df2,
            x='TRANS_DATE',
            y='TransactionValue',
            color='RPTOWNER_RELATIONSHIP',
            hover_data={
                'TRANS_DATE': '|%Y-%m-%d',
                'TransactionValue': ':.2s'
            }
        )
        for trace in line_fig.data:
            fig_insiders_trading.add_trace(trace, row=3, col=1)

        fig_insiders_trading.update_layout(
            plot_bgcolor=main_div_style['backgroundColor'],
            paper_bgcolor=main_div_style['backgroundColor'],
            font=dict(color=main_div_style['color']),
            height=800,
            title_text='Insiders Trading',
            legend=dict(
                orientation='h',
                x=0.5,
                y=1.05,
                xanchor='center',
                yanchor='bottom',
                traceorder='normal',
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(title='', showticklabels=True),
            autosize=True
        )

        return fig_stock_prices, fig_insiders_trading, main_div_style, symbol_style, column_style, date_range_style, symbol2_style, date_range2_style

    app.run_server(debug=True, port=32333)

launch_stocks_insiders_exploration_app()
