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


# exploration_layout = html.Div([
#     dbc.Container([
#         html.H1("Stocks & Insiders Activities", className='text-center mb-4'),
#         dbc.Row([
#             # Left Column: Stocks Layout
#             dbc.Col([
#                 html.H4("Stock Prices Controls"),
#                 dbc.Row([
#                     dbc.Col([
#                         html.Label("Choose a symbol:"),
#                         dcc.Dropdown(
#                             id='explore-symbol',
#                             options=[{'label': i, 'value': i} for i in interim_df_stock_prices['SYMBOL'].unique()],
#                             value='AAPL',
#                             clearable=False,
#                             style={'backgroundColor': '#ffffff', 'color': 'white'}  
#                         ),
#                     ], width=6),
#                     dbc.Col([
#                         html.Label("Choose a column:"),
#                         dcc.Dropdown(
#                             id='explore-column',
#                             options=[
#                                 {'label': 'Low', 'value': 'Low'},
#                                 {'label': 'High', 'value': 'High'},
#                                 {'label': 'Close', 'value': 'Close'},
#                                 {'label': 'Open', 'value': 'Open'}
#                             ],
#                             value='Low',
#                             clearable=False,
#                             style={'backgroundColor': '#ffffff', 'color': 'white'} 
#                         ),
#                     ], width=6),
#                 ]),
#                 dbc.Row([
#                     dbc.Col([html.Label("Choose a date range:")], width=2),
#                     dbc.Col([
#                         dcc.DatePickerRange(
#                             id='explore-date_range',
#                             start_date=interim_df_stock_prices[interim_df_stock_prices['Date'].dt.year == 2014]['Date'].min(),
#                             end_date=interim_df_stock_prices[interim_df_stock_prices['Date'].dt.year == 2014]['Date'].max(),
#                             style={'backgroundColor': '#ffffff', 'color': 'white'}
#                         ),
#                     ], width=4),
#                     dbc.Col([
#                         dcc.Checklist(
#                             id='explore-theme-toggle',
#                             options=[{'label': 'Dark Mode', 'value': 'dark'}],
#                             value=[]
#                         ),
#                     ], width=3),
#                     dbc.Col([
#                         dcc.RadioItems(
#                             id='explore-display_mode',
#                             options=[
#                                 {'label': 'Scatter Points', 'value': 'scatter'},
#                                 {'label': 'Lines', 'value': 'lines'},
#                                 {'label': 'Both', 'value': 'both'}
#                             ],
#                             value='both',
#                             inline=True
#                         )
#                     ], width=3)
#                 ]),
#                 dcc.Loading(
#                     id="loading-explore-graph",  # Add a unique ID to the loading spinner
#                     children=dcc.Graph(id='explore-stock_prices', config={'responsive': True}),
#                     type="circle"  # Type of the spinner (circle is a popular choice)
#                 )
#             ], width=12, lg=6),
#             # Right Column: Insiders Layout
#             dbc.Col([
#                 html.H4("Insiders Trading Controls"),
#                 html.Label("Choose a symbol:"),
#                 dcc.Dropdown(
#                     id='explore-symbol2',
#                     options=[{'label': i, 'value': i} for i in interim_df_merged['ISSUERTRADINGSYMBOL'].unique()],
#                     value='AAPL',
#                     clearable=False,
#                     style={'backgroundColor': '#ffffff', 'color': 'white'}
#                 ),
#                 html.Label("Choose a date range:"),
#                 dcc.DatePickerRange(
#                     id='explore-date_range2',
#                     start_date=interim_df_merged[interim_df_merged['TRANS_DATE'].dt.year == 2014]['TRANS_DATE'].min(),
#                     end_date=interim_df_merged[interim_df_merged['TRANS_DATE'].dt.year == 2014]['TRANS_DATE'].max(),
#                     style={'backgroundColor': '#ffffff', 'color': 'white'}
#                 ),
#                 dcc.Loading(
#                     id="loading-explore-insiders",  # Another unique ID for the insiders graph spinner
#                     children=dcc.Graph(id='explore-insiders_trading', config={'responsive': True}),
#                     type="circle"
#                 )
#             ], width=12, lg=6)
#         ])
#     ], fluid=True)
# ], id='explore-main-div', style={'backgroundColor': '#f8f9fa'})
exploration_layout = html.Div([
    dbc.Container([
        html.H1("Stocks & Insiders Activities", className='text-center my-4'),

        # Informational Text Box for the Page
        dbc.Alert(
            [
                html.H4("How to Use This Page", className="alert-heading"),
                html.P(
                    "Explore stock prices and insider trading activities by selecting options below. "
                    "Use the controls to customize the data displayed in the graphs.",
                    className="mb-0",
                ),
            ],
            color="info",
            dismissable=True,
            is_open=True,
            className="mb-4",
        ),


        dbc.Row([
            # Left Column: Stocks Layout
            dbc.Col([
                html.H4([html.I(className="fas fa-chart-line me-2"),"Stock Prices Controls"]),

                # Informational Text Box for Stock Prices
                dbc.Alert(
                    [
                        html.P(
                            "Select a stock symbol, column (e.g., 'Close', 'Open'), and date range to visualize stock prices. "
                            "You can also choose the display mode and toggle dark mode for better viewing.",
                            className="mb-0",
                        ),
                    ],
                    color="light",
                    dismissable=True,
                    is_open=True,
                    className="mb-3",
                ),

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
                html.H4([html.I(className="fas fa-chart-line me-2"),"Insiders Trading Controls"]),


                # Informational Text Box for Insiders Trading
                dbc.Alert(
                    [
                        html.P(
                            "Select a stock symbol and date range to view insider trading activities. "
                            "This can provide insights into the actions of company insiders.",
                            className="mb-0",
                        ),
                    ],
                    color="light",
                    dismissable=True,
                    is_open=True,
                    className="mb-3",
                ),


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





# prediction_layout = html.Div([
#     dbc.Container([
#         html.H1("Stocks & Insiders Predictions", className='text-center mb-4'),
#         dbc.Row([
#             # Left Column: Stocks Layout
#             dbc.Col([
#                 html.H4("Stock Prices Controls"),
#                 dbc.Row([
#                     dbc.Col([
#                         html.Label("Choose a symbol:"),
#                         dcc.Dropdown(
#                             id='predict-symbol',
#                             options=[{'label': i, 'value': i} for i in processed_df_stock_prices['SYMBOL'].unique()],
#                             value='AAPL',
#                             clearable=False,
#                             style={'backgroundColor': '#ffffff', 'color': 'black'}  
#                         ),
#                     ], width=6),
#                     dbc.Col([
#                         html.Label("Choose a column:"),
#                         dcc.Dropdown(
#                             id='predict-column',
#                             options=[
#                                 {'label': 'Low', 'value': 'Low'},
#                                 {'label': 'High', 'value': 'High'},
#                                 {'label': 'Close', 'value': 'Close'},
#                                 {'label': 'Open', 'value': 'Open'}
#                             ],
#                             value='Low',
#                             clearable=False,
#                             style={'backgroundColor': '#ffffff', 'color': 'black'}  
#                         ),
#                     ], width=6),
#                 ]),
#                 dbc.Row([
#                     dbc.Col([html.Label("Choose a date range:")], width=2),
#                     dbc.Col([
#                         dcc.DatePickerRange(
#                             id='predict-date_range',
#                             start_date=processed_df_stock_prices[processed_df_stock_prices['Date'].dt.year == 2014]['Date'].min(),
#                             end_date=processed_df_stock_prices[processed_df_stock_prices['Date'].dt.year == 2017]['Date'].max(),
#                             style={'backgroundColor': '#ffffff', 'color': 'black'}
#                         ),
#                     ], width=4),
#                     dbc.Col([
#                         dcc.Checklist(
#                             id='predict-theme-toggle',
#                             options=[{'label': 'Dark Mode', 'value': 'dark'}],
#                             value=[]
#                         ),
#                     ], width=3)
#                 ]),
#                 dcc.Loading(
#                     id="loading-predict-graph",  # Unique ID for the prediction spinner
#                     children=dcc.Graph(id='predict-stock_prices', config={'responsive': True}),
#                     type="circle"
#                 )
#             ], width=12, lg=6,style={'padding': '0px', 'width': '100%'}),
#         ],justify='center',style={'padding': '0px', 'width': '100%'})
#     ], fluid=True)
# ], id='predict-main-div', style={'padding': '0px', 'width': '100%','backgroundColor': '#f8f9fa'})
prediction_layout = html.Div([
    dbc.Container([
        html.H1("Stocks & Insiders Predictions", className='text-center my-4'),

        # Informational Text Box for the Page
        dbc.Alert(
            [
                html.H4("How to Use This Page", className="alert-heading"),
                html.P(
                    "View predicted stock prices based on historical data and insider trading activities. "
                    "Use the controls below to customize the predictions displayed in the graph.",
                    className="mb-0",
                ),
            ],
            color="info",
            dismissable=True,
            is_open=True,
            className="mb-4",
        ),

        dbc.Row([
            # Left Column: Stocks Layout
            dbc.Col([
                html.H4([html.I(className="fas fa-chart-line me-2"),"Stock Prices Controls"]),

                # Informational Text Box for Stock Prices Predictions
                dbc.Alert(
                    [
                        html.P(
                            "Select a stock symbol, column (e.g., 'Close', 'Open'), and date range to view predicted stock prices. "
                            "You can also toggle dark mode for better viewing.",
                            className="mb-0",
                        ),
                    ],
                    color="light",
                    dismissable=True,
                    is_open=True,
                    className="mb-3",
                ),

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
        ],justify='center',style={'padding': '0px 12px', 'width': '100%'})
    ], fluid=True)
], id='predict-main-div', style={'padding': '0px', 'width': '100%','backgroundColor': '#f8f9fa'})




# Footer Layout
footer = dbc.Container([
    html.Hr(),
    html.P("© 2024 Stocks & Insiders App", className='text-center')
], style={'textAlign': 'center', 'padding': '1rem 0'})



# # Define the landing page layout
landing_layout = html.Div([
    dbc.Container([
        html.H1("Welcome to the Stocks & Insiders Prediction App of the MPA Platform", className='text-center my-4'),
        
        # Quick Project Summary
        dbc.Card([
            dbc.CardHeader(html.H2("Project Summary")),
            dbc.CardBody([
                html.P(
                    "Predicting stock prices is a challenging task due to the inherent volatility and complexity of financial markets. "
                    "By leveraging insider trading data and stock prices, we aim to enhance investment strategies and provide valuable insights into the impact of insider trading on stock prices.",
                    className="lead",
                ),
                html.P(
                    "This project serves as a starting point to predict 'Low', 'High', 'Open', 'Close', and 'Volume' stock prices. "
                    "We use two datasets: an insider trading dataset from the SEC and stock price data from Kaggle.",
                ),
                dbc.Button("Explore Data", color="primary", href="/data_exploration", className="mt-2"),
            ])
        ], className="mb-4"),
        
        # Project Terms and Jargon
        dbc.Card([
            dbc.CardHeader(html.H2("Project Terms and Jargon")),
            dbc.CardBody([
                html.Ul([
                    html.Li([
                        html.B("Insider Trading: "),
                        "The buying or selling of a company's stock by individuals who have access to private, non-public information about the company."
                    ]),
                    html.Li([
                        html.B("Ticker Symbol: "),
                        "A unique abbreviation for a company's stock on the market (e.g., 'AAPL' for Apple Inc.)."
                    ]),
                    html.Li([
                        html.B("Transaction Type: "),
                        "Indicates whether the insider is buying or selling stock."
                    ]),
                    html.Li([
                        html.B("Price: "),
                        "The price at which the insider bought or sold the stock."
                    ]),
                    html.Li([
                        html.B("Volume: "),
                        "The total number of shares traded in a particular stock during a certain time period."
                    ]),
                    html.Li([
                        html.B("Stock Price Data: "),
                        "Includes 'Open', 'Close', 'Low', 'High', and 'Volume' metrics for stocks."
                    ]),
                    html.Li([
                        html.B("Price Impact: "),
                        "The effect that a transaction has on the stock price."
                    ]),
                ]),
            ])
        ], className="mb-4"),

        # Link to README
        dbc.Card([
            dbc.CardHeader(html.H2("Project Documentation")),
            dbc.CardBody([
                html.P(
                    "For detailed information about the project structure, data processing, and analysis, please refer to our README file.",
                ),
                dbc.Button("View README", color="secondary", href="https://github.com/OSawani/MarketPulseAnalytics", target="_blank"),
            ])
        ], className="mb-4"),

        # Business Requirements
        dbc.Card([
            dbc.CardHeader(html.H2("Business Requirements")),
            dbc.CardBody([
                html.Ol([
                    html.Li("Analyze the impact of insider trading on stock trading volume."),
                    html.Li("Predict stock metrics ('Low', 'High', 'Open', 'Close') using historical stock data."),
                    html.Li("Improve volume predictions by incorporating insider trading data and engineered features."),
                    html.Li("Develop interactive dashboards for data visualization and model predictions."),
                ]),
                dbc.Button("View Predictions", color="success", href="/predictions", className="mt-2"),
            ])
        ], className="mb-4"),

        # Project Hypotheses and Validation
        dbc.Card([
            dbc.CardHeader(html.H2("Project Hypotheses and Validation")),
            dbc.CardBody([
                html.Ul([
                    html.Li([
                        html.B("Hypothesis 1: "),
                        "Using only the stocks dataset, stock metrics can be predicted using linear approximation without additional features.",
                        html.Br(),
                        html.B("Conclusion: "),
                        "Linear regression models provided low error and good R² scores for 'Low', 'High', 'Open', and 'Close' prices, but not for 'Volume'."
                    ]),
                    html.Li([
                        html.B("Hypothesis 2: "),
                        "Including insider activity data will improve the prediction of 'Volume' and ignoring it will negatively affect model performance.",
                        html.Br(),
                        html.B("Conclusion: "),
                        "Incorporating insider features sometimes improved RMSE for 'Volume', but the negative R² scores indicated insufficient features or inappropriate models."
                    ]),
                    html.Li([
                        html.B("Hypothesis 3: "),
                        "Using XGBoost can improve 'Volume' regression predictions compared to linear regression.",
                        html.Br(),
                        html.B("Conclusion: "),
                        "XGBoost improved RMSE and R² scores for 'Volume' predictions, capturing non-linear relationships."
                    ]),
                    html.Li([
                        html.B("Hypothesis 4: "),
                        "The 'Volume' prediction can be further improved by incorporating insider trading features and engineered features.",
                        html.Br(),
                        html.B("Conclusion: "),
                        "Including engineered features and insider data further improved 'Volume' prediction accuracy."
                    ]),
                ]),
            ])
        ], className="mb-4"),

        # Call to Action Buttons
        dbc.Row([
            dbc.Col([
                dbc.Button("Explore Data", color="primary", href="/data_exploration", className="me-2"),
                dbc.Button("View Predictions", color="success", href="/predictions"),
            ], className="text-center"),
        ], className="my-4"),
    ], fluid=True),
], style={'backgroundColor': '#f8f9fa'})




# # Navbar with Home link and updated styling
navbar = dbc.Navbar(
    [
        dbc.Container(
            [
                dbc.NavbarBrand("Market Pulse Analytics", href="/", className="me-auto"),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dcc.Link("Home", href="/", className="nav-link")),
                            dbc.NavItem(dcc.Link("Data Exploration", href="/data_exploration", className="nav-link")),
                            dbc.NavItem(dcc.Link("Predictions", href="/predictions", className="nav-link")),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            fluid=True,
        )
    ],
    color="dark",
    dark=True,
    className="navbar navbar-expand-lg bg-dark",
    style={"padding": "0.5rem 1rem"},
)
# Define the navigation bar
# navbar = dbc.NavbarSimple(
#     children=[
#         dbc.NavItem(dcc.Link("Data Exploration", href="/data_exploration", className="nav-link")),
#         dbc.NavItem(dcc.Link("Predictions", href="/predictions", className="nav-link")),
#     ],
#     brand="Stocks & Insiders App",
#     color="primary",
#     dark=True,
# )




# Define the content placeholder
content = html.Div(id="page-content", style={'flex': '1'})
# # Define the content placeholder
# content = html.Div(id="page-content")



# Set up the app layout
app_layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    content,
    footer
], style={
    'display': 'flex',
    'flexDirection': 'column',
    'minHeight': '100vh',  # Ensures the parent div takes up at least the full viewport height
    'backgroundColor': '#f8f9fa',
})
# # Set up the app layout
# app_layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    content
])