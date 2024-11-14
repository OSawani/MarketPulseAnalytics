from dash import html, dcc
import dash_bootstrap_components as dbc
from shared_variables import interim_df_merged, interim_df_stock_prices, interim_df_insider_transactions, processed_df_stock_prices



def get_component_style(theme):
    if False:#'dark' in theme:
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



exploration_layout = html.Div([
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
                            id='explore-symbol',
                            options=[{'label': i, 'value': i} for i in interim_df_stock_prices['SYMBOL'].unique()],
                            value='AAPL',
                            clearable=False,
                            style={'backgroundColor': '#ffffff', 'color': 'white'}  
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Choose a column:"),
                        dcc.Dropdown(
                            id='explore-column',
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
                    dbc.Col([html.Label("Choose a date range:")], width=2),
                    dbc.Col([
                        dcc.DatePickerRange(
                            id='explore-date_range',
                            start_date=interim_df_stock_prices[interim_df_stock_prices['Date'].dt.year == 2014]['Date'].min(),
                            end_date=interim_df_stock_prices[interim_df_stock_prices['Date'].dt.year == 2014]['Date'].max(),
                            style={'backgroundColor': '#ffffff', 'color': 'white'}
                        ),
                    ], width=4),
                    dbc.Col([
                        dcc.Checklist(
                            id='explore-theme-toggle',
                            options=[{'label': 'Dark Mode', 'value': 'dark'}],
                            value=[]
                        ),
                    ], width=3),
                    dbc.Col([
                        dcc.RadioItems(
                            id='explore-display_mode',
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
                dcc.Loading(
                    id="loading-explore-graph",  # Add a unique ID to the loading spinner
                    children=dcc.Graph(id='explore-stock_prices', config={'responsive': True}),
                    type="circle"  # Type of the spinner (circle is a popular choice)
                )
            ], width=12, lg=6),
            # Right Column: Insiders Layout
            dbc.Col([
                html.H4("Insiders Trading Controls"),
                html.Label("Choose a symbol:"),
                dcc.Dropdown(
                    id='explore-symbol2',
                    options=[{'label': i, 'value': i} for i in interim_df_merged['ISSUERTRADINGSYMBOL'].unique()],
                    value='AAPL',
                    clearable=False,
                    style={'backgroundColor': '#ffffff', 'color': 'white'}
                ),
                html.Label("Choose a date range:"),
                dcc.DatePickerRange(
                    id='explore-date_range2',
                    start_date=interim_df_merged[interim_df_merged['TRANS_DATE'].dt.year == 2014]['TRANS_DATE'].min(),
                    end_date=interim_df_merged[interim_df_merged['TRANS_DATE'].dt.year == 2014]['TRANS_DATE'].max(),
                    style={'backgroundColor': '#ffffff', 'color': 'white'}
                ),
                dcc.Loading(
                    id="loading-explore-insiders",  # Another unique ID for the insiders graph spinner
                    children=dcc.Graph(id='explore-insiders_trading', config={'responsive': True}),
                    type="circle"
                )
            ], width=12, lg=6)
        ])
    ], fluid=True)
], id='explore-main-div', style={'backgroundColor': '#f8f9fa'})


prediction_layout = html.Div([
    dbc.Container([
        html.H1("Stocks & Insiders Predictions", className='text-center mb-4'),
        dbc.Row([
            # Left Column: Stocks Layout
            dbc.Col([
                html.H4("Stock Prices Controls"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Choose a symbol:"),
                        dcc.Dropdown(
                            id='predict-symbol',
                            options=[{'label': i, 'value': i} for i in processed_df_stock_prices['SYMBOL'].unique()],
                            value='AAPL',
                            clearable=False,
                            style={'backgroundColor': '#ffffff', 'color': 'black'}  
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Choose a column:"),
                        dcc.Dropdown(
                            id='predict-column',
                            options=[
                                {'label': 'Low', 'value': 'Low'},
                                {'label': 'High', 'value': 'High'},
                                {'label': 'Close', 'value': 'Close'},
                                {'label': 'Open', 'value': 'Open'}
                            ],
                            value='Low',
                            clearable=False,
                            style={'backgroundColor': '#ffffff', 'color': 'black'}  
                        ),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([html.Label("Choose a date range:")], width=2),
                    dbc.Col([
                        dcc.DatePickerRange(
                            id='predict-date_range',
                            start_date=processed_df_stock_prices[processed_df_stock_prices['Date'].dt.year == 2014]['Date'].min(),
                            end_date=processed_df_stock_prices[processed_df_stock_prices['Date'].dt.year == 2017]['Date'].max(),
                            style={'backgroundColor': '#ffffff', 'color': 'black'}
                        ),
                    ], width=4),
                    dbc.Col([
                        dcc.Checklist(
                            id='predict-theme-toggle',
                            options=[{'label': 'Dark Mode', 'value': 'dark'}],
                            value=[]
                        ),
                    ], width=3)
                ]),
                dcc.Loading(
                    id="loading-predict-graph",  # Unique ID for the prediction spinner
                    children=dcc.Graph(id='predict-stock_prices', config={'responsive': True}),
                    type="circle"
                )
            ], width=12, lg=6,style={'padding': '0px', 'width': '100%'}),
        ],justify='center',style={'padding': '0px', 'width': '100%'})
    ], fluid=True)
], id='predict-main-div', style={'padding': '0px', 'width': '100%','backgroundColor': '#f8f9fa'})


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
app_layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    content
])