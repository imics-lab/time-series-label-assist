import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Import layouts from app modules
from apps.preprocessing import layout as preprocessing_layout
from apps.manual_labeling import layout as manual_labeling_layout
from apps.model_training import layout as model_training_layout
from apps.prediction import layout as prediction_layout
from apps.correct_autolabels import layout as correct_autolabels_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'ALVI'
app.config.suppress_callback_exceptions = True  # Necessary for multi-page apps

# Define the app layout with navigation tabs
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

# Callback to switch between tabs
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")]
)
def render_tab_content(active_tab):
    if active_tab == "preprocessing":
        return preprocessing_layout
    elif active_tab == "manual_labeling":
        return manual_labeling_layout
    elif active_tab == "model_training":
        return model_training_layout
    elif active_tab == "prediction":
        return prediction_layout
    elif active_tab == "correct_autolabels":
        return correct_autolabels_layout
    return "This tab has not been implemented yet."

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
