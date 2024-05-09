from dash import html

# configurations
# build model
# save model

from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from dash.exceptions import PreventUpdate
# Import your CNN model and any other necessary modules
from prediction import model, split
import flask
import json
import csv
import io
import sys
import dash_bootstrap_components as dbc

def layout():
    if not flask.request:
        return html.Div()
    else:
        return html.Div([
            html.H3('Model Configuration and Training'),
            html.H5('Set Time-Series Data Parameters'),

            dbc.Row([
                dbc.Col([
                    html.Label('Specifications for length of timesteps (e.g., highest_frequency * seconds):'),
                    dcc.Input(id='window-size', type='number', value=96, placeholder='Window Size', className='mb-2'),
                    dcc.Input(id='steps', type='number', value=96, placeholder='Steps', className='mb-2'),
                    dbc.Button('Update Model Training Parameters', id='update-params', className='mb-3'),
                    html.Div(id='param-output', style={'marginBottom': '20px'}),
                ], width=6),
            ], style={'marginBottom': '20px'}),

            dbc.Row([
                dbc.Col([
                    html.H5('Model Saving Options'),
                    html.Div([
                        html.Label('Name your model:'),
                        dcc.Input(id='model-name', type='text', value='1D_CNN', placeholder='Model Name', className='mb-2'),
                        dbc.Button('Build and Train Model', id='build-train', className='mt-2 mb-2'),
                    ]),
                    html.Div(id='model-output', style={'marginBottom': '20px'}),
                ], width=6),
            ], style={'marginBottom': '20px'}),
        ])

@callback(
    Output('param-output', 'children'),
    Input('update-params', 'n_clicks'),
    State('window-size', 'value'),
    State('steps', 'value'),
)
def update_parameters(n_clicks, window_size, steps):
    if n_clicks is None:
        raise PreventUpdate
    # Load and preprocess data here, similar to your notebook
    df = pd.read_csv('storage/manual_label_df.csv')
    labelListDF = pd.read_csv('storage/label_list.csv')
    labelList = list(labelListDF)

    # if has confidence, drop it.
    if "confidence" in df.columns:
        new_df = df.copy().drop('confidence', axis=1).set_index('datetime')
    else:
        new_df=df.copy().set_index('datetime')
    new_np = split.TimeSeriesNP(window_size, steps)
    new_np.setArrays(new_df, encode=True, one_hot_encode=False, labels=labelList)

    # Define a path to the assets directory
    prediction_directory = 'storage'
    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)

    file_path = os.path.join(prediction_directory, 'new_np.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump(new_np, f)

    # Process and show some output about data shapes or status
    return f"Data prepared with x shape: {str(new_np.x.shape)} and y shape: {str(new_np.y.shape)}"

@callback(
    Output('model-output', 'children'),
    Input('build-train', 'n_clicks'),
    State('window-size', 'value'),
    State('steps', 'value'),
    State('model-name', 'value'),
)
def build_and_train_model(n_clicks, window_size, steps, model_name):
    if n_clicks is None:
        raise PreventUpdate
    
    # Define a path to the assets directory
    prediction_directory = 'storage'
    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)

    file_path = os.path.join(prediction_directory, 'new_np.pkl')

    with open(file_path, 'rb') as f:
        new_np = pickle.load(f)

    cnn = model.CNN()
    x_train, x_validation, y_train, y_validation = train_test_split(new_np.x, new_np.y.ravel(), test_size = 0.25)
    cnn.only_train_data(x_train, x_validation, y_train, y_validation)
    cnn.build()
    cnn.train()

    # Define directories for models and dimensions, ensure they use backslashes for Windows paths
    models_dir = 'prediction\\models'
    dimensions_dir = 'prediction\\dimensions'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(dimensions_dir, exist_ok=True)

    # Save the model using backslashes
    model_path = os.path.join(models_dir, f"{model_name}.h5").replace('/', '\\')
    cnn.model.save(model_path)

    # Save dimensions to a CSV using backslashes
    dimensions_path = os.path.join(dimensions_dir, f"{model_name}.csv").replace('/', '\\')
    with open(dimensions_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([window_size, steps])

    config_path = 'config.json'
    # Update config.json with the model and dimensions paths
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
    else:
        config = {}

    config['model_path'] = model_path.replace('\\', '/')
    config['dimensions_path'] = dimensions_path.replace('\\', '/')
    config['can_access_prediction'] = True
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    # Capture the output of model.summary()
    old_stdout = sys.stdout
    sys.stdout = model_summary = io.StringIO()
    cnn.model.summary()
    sys.stdout = old_stdout  # Reset standard output to its original value
    
    model_summary_str = model_summary.getvalue()
    
    return html.Div([
        html.H5(f"Model {model_name} built and trained."),
        html.H6("Model and dimensions saved."),
        html.Pre("Model summary here..."),
        html.Pre(model_summary_str)  # Display model summary
    ])

