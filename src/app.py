import pandas as pd
import numpy as np
import re
import os
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
from sklearn.metrics import root_mean_squared_error,make_scorer,mean_absolute_percentage_error# alternative
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from datetime import timedelta

from shared_variables import interim_df_merged, interim_df_stock_prices, interim_df_insider_transactions, processed_df_stock_prices
from backend import predict_stock_prices
from frontend import exploration_layout, get_component_style, app_layout, prediction_layout
# --------------------------------------------
# --------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
print("App script directory: ", current_dir)
if current_dir.endswith('src'):
    # go two directories up to get to the root directory
    os.chdir(os.path.join(current_dir, '..'))
    print("app new current directory to: ", os.getcwd())
print("----------------------------\n")
# --------------------------------------------




# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = app_layout
server = app.server  # Expose the server for deployment
# --------------------------------------------





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
# --------------------------------------------


# --------------------------------------------
@app.callback(
    Output('predict-stock_prices', 'figure'),
    Output('predict-main-div', 'style'),
    Output('predict-symbol', 'style'),  
    Output('predict-column', 'style'),  
    Output('predict-date_range', 'style'),  
    [Input('predict-symbol', 'value'),
    Input('predict-column', 'value'),
    Input('predict-date_range', 'start_date'),
    Input('predict-date_range', 'end_date'),
    Input('predict-theme-toggle', 'value')]
)
def prediction_update_figure(symbol, column, start_date1, end_date1, theme):

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
    # only if date start and end are not valid we do the rest of the code else return  without doing anything
    if start_date1>end_date1:
        temp=start_date1
        start_date1=end_date1
        end_date1=temp
    fig_stock_prices = predict_stock_prices(processed_df_stock_prices, symbol, start_date1, end_date1, column)
    fig_stock_prices.update_layout(
        plot_bgcolor=main_div_style['backgroundColor'],
        paper_bgcolor=main_div_style['backgroundColor'],
        font=dict(color=main_div_style['color'])
        )
    return fig_stock_prices,main_div_style, symbol_style, column_style, date_range_style

# --------------------------------------------
@app.callback(
    Output('explore-stock_prices', 'figure'),
    Output('explore-insiders_trading', 'figure'),
    Output('explore-main-div', 'style'),
    Output('explore-symbol', 'style'),  
    Output('explore-column', 'style'),  
    Output('explore-date_range', 'style'),  
    Output('explore-symbol2', 'style'),  
    Output('explore-date_range2', 'style'),  
    [Input('explore-symbol', 'value'),
    Input('explore-column', 'value'),
    Input('explore-symbol2', 'value'),
    Input('explore-date_range2', 'start_date'),
    Input('explore-date_range2', 'end_date'),
    Input('explore-date_range', 'start_date'),
    Input('explore-date_range', 'end_date'),
    Input('explore-theme-toggle', 'value'),
    Input('explore-display_mode', 'value')]
)
def exploration_update_figure(symbol, column, symbol2, start_date2, end_date2, start_date1, end_date1, theme, display_mode):
    # Prepare data for stock prices
    df = interim_df_stock_prices[(interim_df_stock_prices['SYMBOL'] == symbol) &
                        (interim_df_stock_prices['Date'] >= start_date1) &
                        (interim_df_stock_prices['Date'] <= end_date1)].copy()
    # Prepare data for insiders trading
    df2 = interim_df_insider_transactions[(interim_df_insider_transactions['ISSUERTRADINGSYMBOL'] == symbol2) &
                                (interim_df_insider_transactions['TRANS_DATE'] >= start_date2) &
                                (interim_df_insider_transactions['TRANS_DATE'] <= end_date2)].copy()
    
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
# --------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=False, port=32335)  # Disabled debug mode for production