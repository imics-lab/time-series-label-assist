import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_player
import os
import json
import importlib

# Import modules
from apps import preprocessing, manual_labeling, model_training, prediction, correct_autolabels

# Track registered modules
registered_modules = set()

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
    # Load the current configuration to check access rights
    config = load_config()

    # Check access for all tabs except 'preprocessing'
    if active_tab != "preprocessing" and not config.get(f"can_access_{active_tab}", False):
        return html.Div(f"You do not have access to the {active_tab.replace('_', ' ').title()} section. Please complete the necessary steps in the previous sections.")

    # If this is the first time this tab is being accessed, register its callbacks
    if active_tab not in registered_modules:
        try:
            # Dynamically access the module from the global namespace
            module = globals()[active_tab]
            # Check if the module has a function to register callbacks
            if hasattr(module, 'register_callbacks'):
                module.register_callbacks(app)
                registered_modules.add(active_tab)
                print(f"Registered callbacks for {active_tab}")
        except KeyError:
            return html.Div("This tab has not been implemented yet.")
    
    # Now that we've confirmed the user has access (or it's the 'preprocessing' tab), get the layout from the module
    layout_func = globals().get(active_tab).layout
    return layout_func()


if __name__ == '__main__':
    app.run_server(debug=True, port=8055)