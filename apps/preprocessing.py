from utilities.data_manager import load_data, process_data
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import datetime
from dash.exceptions import PreventUpdate
import dash_player
import time
import os
import json
import glob as glob
import shutil

def layout():
    layout = html.Div([
        html.H3('Data Preprocessing'), # Title for the section
        html.H5('Load CSV Time-Series Data'),
        dcc.Upload( # Component to upload files
            id='upload-data',
            children=html.Button('Load CSV Data', id='load-CSV-data-button'),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
            multiple=True
        ), 
        html.Div(id='output-data-upload', style={'marginBottom': '20px'}), # Increased margin bottom for spacing
        html.H5('Load CSV Label List'),
        dcc.Upload(  # Upload component for the label CSV
            id='upload-labels',
            children=html.Button('Load Label List', id='load-labels-button'),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
            multiple=False
        ),
        html.Div(id='labels-output', style={'marginBottom': '20px'}),
        html.H5("Configurations"),
        dbc.Row([  # Two columns for selecting time and label columns
            dbc.Col(html.Div([
                html.Label("Select Time Column:"),
                dcc.Dropdown(id='time-column-dropdown'),
            ], style={'marginBottom': '20px'}), width=6), # Added margin bottom
            dbc.Col(html.Div([
                html.Label("Select Label Column:"),
                dcc.Dropdown(id='label-column-dropdown'),
            ], style={'marginBottom': '20px'}), width=6) # Added margin bottom
        ]),
        dbc.Row([  # Row for feature selection and default plot columns with select all functionality
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.Label("Select Feature Columns:"), width=8),
                    dbc.Col(dbc.Button("Select All", id="select-all-features", n_clicks=0, className="mr-1"), width=4),
                ]),
                dbc.Checklist(  # Checklist to select features
                    options=[],
                    id='feature-columns-checkboxes',
                    inline=True,
                    switch=True,
                    style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fill, minmax(120px, 1fr))', 'overflowY': 'auto', 'maxHeight': '200px'}
                ),
            ], width=6, style={'marginBottom': '20px'}), # Added margin bottom
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.Label("Select Features to Plot by Default:"), width=8),
                    dbc.Col(dbc.Button("Select All", id="select-all-plots", n_clicks=0, className="mr-1"), width=4),
                ]),
                dbc.Checklist( # Checklist to select default plot columns
                    options=[],
                    id='plot-columns-checkboxes',
                    inline=True,
                    switch=True,
                    style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fill, minmax(120px, 1fr))', 'overflowY': 'auto', 'maxHeight': '200px'}
                )
            ], width=6, style={'marginBottom': '20px'}) # Added margin bottom
        ]),
        html.H5("Load Video Data", style={'marginBottom': '20px'}),
        dcc.Upload(
            id='upload-video',
            children=html.Button('Load Video', id='load-video-button'),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px', 'marginBottom': '20px'},
            multiple=False  # Assuming only one video file is needed at a time
        ),
        html.H5("Video", style={'marginBottom': '20px'}),
        # Initialize DashPlayer directly in the layout
        dash_player.DashPlayer(
            id='video-player',
            controls=True,
            playing=False,
            width='100%',
            height='400px',
            url="init",  # Initial empty URL or placeholder video URL
            style={'marginBottom': '20px'},
        ), 
        html.H5("Video Data Sync Offset", style={'marginBottom': '10px'}),
        html.Div([
            html.P([html.Strong("Zero Offset:"), " Sync Start - Video and data begin together."]),
            html.P([html.Strong("Positive Offset:"), " Video Delay - Start video [offset] seconds after data."]),
            html.P([html.Strong("Negative Offset:"), " Data Delay - Start data [offset] seconds after video."]),
            dbc.Input(id='video-data-offset', type='number', placeholder="Enter offset in seconds...", step=1, style={'marginBottom': '10px'}),
        ], style={'marginBottom': '20px'}),
        dbc.Button("Save Configuration", id='save-config-button', className="mb-3", style={'marginTop': '20px', 'marginBottom': '20px'}),
        html.Div(id='config-save-status', style={'marginTop': '20px'}),
        dcc.Store(id='stored-manual-df'),
        dcc.Store(id='video-data'),
        dcc.Store(id='stored-labels')
    ])
    return layout

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(['Unsupported file format']), None
        display = html.Div([
            html.H5(filename),
            html.P(f"First 5 rows of {filename}:"),
            dash_table.DataTable(
                data=df.head().to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'}
            )
        ])
        return display, df.to_json(date_format='iso', orient='split')
    except Exception as e:
        return html.Div([f'There was an error processing this file: {e}']), None

@callback(
    Output('trigger', 'children'),
    Input('tabs', 'active_tab'),
    prevent_initial_call=True
)
def set_trigger(active_tab):
    return 'Loaded'

# Callback to parse and display the contents of the uploaded files
@callback(
    [Output('output-data-upload', 'children'),
     Output('stored-manual-df', 'data')], # Updating the display and storing the data
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    prevent_initial_call=False
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = []
        data_store = None
        for c, n, d in zip(list_of_contents, list_of_names, list_of_dates):
            parsed_content, df_json = parse_contents(c, n, d)
            children.append(parsed_content)
            if df_json:
                data_store = df_json
        return children, data_store
    else:
        raise PreventUpdate # Prevent updating if no files are uploaded

# Callback to parse and display the contents of the uploaded labels file
@callback(
    [Output('labels-output', 'children'),
     Output('stored-labels', 'data')],
    [Input('upload-labels', 'contents')],
    [State('upload-labels', 'filename')],
    prevent_initial_call=False
)
def update_labels(contents, filename):
    if contents is None:
        raise PreventUpdate
    decoded = base64.b64decode(contents.split(',')[1])
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return html.Div([
        html.H5(filename),
        html.Pre(df.to_csv(index=False))
    ]), df.to_json(date_format='iso', orient='split')

# Callbacks to update the column selection dropdowns based on the data in the store
@callback(
    [Output('time-column-dropdown', 'options'),
     Output('label-column-dropdown', 'options'),
     Output('feature-columns-checkboxes', 'options'),
     Output('plot-columns-checkboxes', 'options')],
    Input('stored-manual-df', 'data'),
    prevent_initial_call=False
)
def update_column_dropdowns(jsonified_dataframe):
    if jsonified_dataframe is None:
        raise PreventUpdate
    df = pd.read_json(jsonified_dataframe, orient='split')
    columns = [{'label': col, 'value': col} for col in df.columns]
    return columns, columns, columns, columns

# Callback to process selected data based on user input
@callback(
    Output('processed-data-store', 'data'),
    [Input('time-column-dropdown', 'value'),
     Input('label-column-dropdown', 'value'),
     Input('feature-columns-checkboxes', 'value')],
    State('stored-manual-df', 'data'),
    prevent_initial_call=False
)
def process_data(time_col, label_col, feature_cols, jsonified_dataframe):
    if jsonified_dataframe is None:
        raise PreventUpdate
    df = pd.read_json(jsonified_dataframe, orient='split')
    
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        df = df.sort_values(by=[time_col]).reset_index(drop=True)
        time_col_first = [time_col] + [col for col in df.columns if col != time_col]
        df = df[time_col_first]

    if label_col and feature_cols:
        df = df[[time_col, label_col] + feature_cols]
    
    return df.to_json(date_format='iso', orient='split')

# Callbacks to manage 'Select All' functionality for checkboxes
@callback(
    Output('feature-columns-checkboxes', 'value'),
    Input('select-all-features', 'n_clicks'),
    State('feature-columns-checkboxes', 'options'),
    prevent_initial_call=True
)
def select_all_features(n_clicks, options):
    if n_clicks % 2 == 1:
        return [option['value'] for option in options]
    else:
        return []

@callback(
    Output('plot-columns-checkboxes', 'value'),
    Input('select-all-plots', 'n_clicks'),
    State('plot-columns-checkboxes', 'options'),
    prevent_initial_call=True
)
def select_all_plots(n_clicks, options):
    if n_clicks % 2 == 1:
        return [option['value'] for option in options]
    else:
        return []

# Video Callback
# Callback to handle video uploads and manage the video file
@callback(
    Output('video-player', 'url'),  # Updating the URL of the existing video player component
    Input('upload-video', 'contents'),
    State('upload-video', 'filename'),
    prevent_initial_call=False
)
def update_video_output(contents, filename):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    _, file_extension = os.path.splitext(filename)
    
    # Create a consistent naming for the video file
    video_filename = f"manual_video_{filename}"
    video_path = os.path.join('assets', video_filename)
    
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Check if a video file exists and remove it
    existing_files = glob.glob(os.path.join('assets', 'manual_video_*'))
    for file in existing_files:
        os.remove(file)
    
    with open(video_path, 'wb') as f:
        f.write(decoded)
    
    return video_path

# Callback to save configuration
@callback(
    Output('config-save-status', 'children'),
    [Input('save-config-button', 'n_clicks')],
    [State('feature-columns-checkboxes', 'value'),
     State('plot-columns-checkboxes', 'value'),
     State('feature-columns-checkboxes', 'options'),
     State('stored-manual-df', 'data'),
     State('stored-labels', 'data'),
     State('video-player', 'url'),
     State('video-data-offset', 'value')],
    prevent_initial_call=False
)
def save_configuration(n_clicks, selected_features, plot_features, all_options, stored_df_json, stored_labels_json, video_url, offset):
    if n_clicks is None:
        raise PreventUpdate
    
    # Check if the DataFrame is available
    if stored_df_json is None:
        return "No data to save."
    
    # Load dataframes from the stored JSON
    df = pd.read_json(stored_df_json, orient='split')
    labels_df = pd.read_json(stored_labels_json, orient='split')

    # Determine features to omit by comparing all options against selected features
    all_features = [option['value'] for option in all_options]
    features_to_omit = [feature for feature in all_features if feature not in selected_features]

    # Load existing configuration from file
    config_path = 'assets/config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
    else:
        config = {}

    # Update configuration with new values
    config.update({
        "valid_features": selected_features,
        "features_to_omit": features_to_omit,
        "features_to_plot": plot_features,
        "video_path": video_url,
        "offset_manual": offset,
        "preprocessing_completed": True  # Set preprocessing to completed
    })

    # Save the updated configuration back to the JSON file
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)
    
    # Save the DataFrame and labels to CSV files
    df_path = os.path.join('assets', 'manual_label_df.csv')
    labels_path = os.path.join('assets', 'label_list.csv')
    df.to_csv(df_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    return "Configuration and data saved successfully!"
