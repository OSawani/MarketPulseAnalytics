# index.py
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from app import app
from data_exploration_app import layout as exploration_layout
from final_app_prediction import layout as prediction_layout

# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dcc.Link("Data Exploration", href="/data_exploration", className="nav-link")),  # Updated href
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

# Expose the server for deployment
server = app.server  # Added server exposure

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

if __name__ == '__main__':
    app.run_server(debug=False)  # Disabled debug mode
