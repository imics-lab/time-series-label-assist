from utilities.data_manager import load_data, process_data
from dash import html

# Example usage within a tab layout
layout = html.Div([
    html.H3('Data Preprocessing'),
    html.Button('Load Data', id='load-data-button'),
    html.Div(id='data-output')
])

# You would then define a callback to load and process the data when the button is clicked.
