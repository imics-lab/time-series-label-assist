import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_player
import os
import json
import importlib

# Import layouts from app modules
from apps.preprocessing import layout as preprocessing_layout
from apps.manual_labeling import layout as manual_labeling_layout
from apps.model_training import layout as model_training_layout
from apps.prediction import layout as prediction_layout
from apps.correct_autolabels import layout as correct_autolabels_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'ALVI'
app.config.suppress_callback_exceptions = True  # Necessary for multi-page apps

def initialize_config():
    config_path = 'assets/config.json'
    default_config = {
        "can_access_manual_labeling": False,
        "can_access_model_training": False,
        "can_access_prediction": False,
        "can_access_correct_autolabels": False
    }
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        config = default_config
    else:
        with open(config_path, 'r') as file:
            existing_config = json.load(file)
        # Merge defaults with existing to ensure all keys are present
        config = {**default_config, **existing_config}
    
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    return config

# Load the configuration once and use throughout
config = initialize_config()

app.layout = html.Div([
    dbc.Tabs([
        dbc.Tab(label="Preprocessing", tab_id="preprocessing"),
        dbc.Tab(label="Manual Labeling", tab_id="manual_labeling"),
        dbc.Tab(label="Model Training", tab_id="model_training"),
        dbc.Tab(label="Prediction", tab_id="prediction"),
        dbc.Tab(label="Correct Autolabels", tab_id="correct_autolabels")
    ], id="tabs", active_tab="preprocessing"),
    html.Div(id="tab-content", className="p-4")
])

# Load the configuration
def load_config():
    config_path = 'assets/config.json'
    with open(config_path, 'r') as file:
        return json.load(file)

# Define the application structure and callback
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")]
)
def render_tab_content(active_tab):
    config = load_config()  # Load the configuration each time the tab changes

    # Define a dictionary to map tab ids to their respective layout functions
    tab_layouts = {
        "preprocessing": preprocessing_layout,  # Assuming preprocessing is always accessible
        "manual_labeling": manual_labeling_layout,
        "model_training": model_training_layout,
        "prediction": prediction_layout,
        "correct_autolabels": correct_autolabels_layout
    }

    # Check configuration to decide access
    if active_tab != "preprocessing" and not config.get(f"can_access_{active_tab}", False):
        return html.Div(f"You do not have access to the {active_tab.replace('_', ' ').title()} section. Please complete the necessary steps in the previous sections.")

    # Get the layout function based on the active tab, or show a default message if the tab isn't implemented
    layout_func = tab_layouts.get(active_tab, lambda: "This tab has not been implemented yet.")
    return layout_func()  # Call the function to get the layout


if __name__ == '__main__':
    app.run_server(debug=True, port=8055)