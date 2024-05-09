
# interface to correct these labels

from dash import html
import flask
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots
import dash_bootstrap_components as dbc
from dash_bootstrap_components import Row, Col, Button, Tooltip, InputGroup, Form
import dash
from dash import ctx
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import load_figure_template
import dash_player
from dash.exceptions import PreventUpdate
from dash import callback_context
# imports
import math
import os
import glob
import json
from datetime import datetime, timedelta

import pandas as pd
from pandas import Timestamp
import numpy as np
import pickle
from datetime import datetime

from IPython.display import display
import ipywidgets as widgets

# HEAVY LIFTING
import tensorflow as tf
from tensorflow.keras.models import load_model

from prediction import split, model
import umap

############################################################################################################################################################
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
############################################################################################################################################################

# Function to create the main layout
def create_main_layout(conf_thresh, labelList, plotly_umap, lineGraph, graph1, graph2, graph3, video_path):
    return dbc.Container([
        html.H1("Correct Auto Labeled Data"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='umap-graph', figure=plotly_umap), width=8),
            dbc.Col([
                html.Div([
                    dcc.Markdown("**Adjust Parameters for UMAP:**"),
                    html.Div([
                        dbc.Label("Number of Neighbors:", html_for="n_neighbors_input"),
                        dbc.Input(id='n_neighbors_input', type='number', value=15, min=2),
                    ], className="mb-3"),
                    html.Div([
                        dbc.Label("Minimum Distance:", html_for="min_dist_slider"),
                        dcc.Slider(
                            id='min_dist_slider',
                            min=0.01,
                            max=0.99,
                            step=0.01,
                            value=0.1,
                            marks={i / 100: f"{i / 100:.2f}" for i in range(10, 100, 10)}
                        ),
                    ], className="mb-3"),
                    html.Div([
                        dbc.Label("Confidence Threshold:", html_for="confidence_threshold_slider"),
                        dcc.Slider(
                            id='confidence_threshold_slider',
                            min=0.01,
                            max=0.99,
                            step=0.01,
                            value=conf_thresh,
                            marks={i / 100: f"{i / 100:.2f}" for i in range(0, 100, 10)}
                        ),
                    ], className="mb-3"),
                    dbc.Button("Update UMAP", id='submit_button', n_clicks=0, className="btn-primary"),
                ]),
                html.Div([
                    dcc.Markdown("**Change Label**"),
                    dcc.Dropdown(labelList, '', id='dropdown', style={'width': '100%'}),
                    dbc.Button('Add Label', id='button', n_clicks=0, className="mt-2")
                ], className="mt-4"),
            ], width=4)
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Start Time:", className="font-weight-bold"),
                dcc.Input(id='ts-start-input', type='text', placeholder="YYYY-MM-DD HH:MM:SS"),
                html.Div(id='start-output')
            ], width=3),
            dbc.Col([
                dbc.Label("End Time:", className="font-weight-bold"),
                dcc.Input(id='ts-end-input', type='text', placeholder="YYYY-MM-DD HH:MM:SS"),
                html.Div(id='end-output')
            ], width=3),
            dbc.Col([
                dbc.Label("Change Label:", className="font-weight-bold"),
                dcc.Dropdown(labelList, placeholder='Select a Label', id='ts-label-selection'),
                html.Div(id='label-output')
            ], width=3),
            dbc.Col([
                dbc.Label("Degree of Confidence:", className="font-weight-bold"),
                dcc.Dropdown(["High", "Medium", "Low", "Undefined"], placeholder='Select a Confidence Level', id='ts-confidence-selection'),
                html.Div(id='confidence-output')
            ], width=3),
        ]),
        dbc.Checklist(
            id='ts-fill-ui-checkbox',
            options=[{'label': 'Fill Start/End Time with Selected Range', 'value': 'fill-ui'}],
            value=[],
            className="mt-3"
        ),
        dbc.Button('Add Label', id='btn-manual-label', n_clicks=0, className="mt-3"),
        html.Label('Time-Series Plot', className="font-weight-bold"),
        dcc.Graph(id='plot-clicked', figure=lineGraph),
        dbc.Row([
            dbc.Col(dcc.Graph(id='graph1', figure=graph1), width=4),
            dbc.Col(dcc.Graph(id='graph2', figure=graph2), width=4),
            dbc.Col(dcc.Graph(id='graph3', figure=graph3), width=4),
            dcc.Store(id='store_data', data = None, storage_type='memory'),
        ]),
        html.Label('Video', className="font-weight-bold"),
        dash_player.DashPlayer(
            id='umap-video-player',
            url=video_path,
            controls=True,
            width='100%',
            height='400px',
        ),
        dbc.Button('Sync Video to Other Visualizations', id='vid_sync_button', n_clicks=0, className="mt-3"),
        dbc.Button('Save Changes', id='save_changes_button', n_clicks=0, className="btn-danger mt-3 mb-3"),
        dbc.Tooltip("Click to save all changes. This should be the last operation after all edits.", target="save_changes_button"),
        html.Div(id='save_status'),
    ], fluid=True)

# Main layout function with request check
def layout():
    if flask.has_request_context():
        global new_model, new_np, orig_df, df, labelList, video_start_time, config, window_size, step, testModel, label_num_dict, colorList, colorDict, base_color_map, num_to_label_dict
        global cols, valid_features, offset, features_to_omit, windows, conf_thresh, embedding_df, embedding, umap_to_ts_mapping, predicted_labels, predictions, npInput, npInput_labels
        global modified_indices

        working_dir = os.getcwd()

        # necessary data loaded
        # 1. model_select.value
        # Path to the assets directory
        assets_dir = os.path.join(working_dir, "assets")
        storage_dir = os.path.join(working_dir, "storage")
        # Search for .h5 files in the assets directory
        model_files = glob.glob(os.path.join(storage_dir, '*.h5'))

        # Check if any .h5 files were found
        if model_files:
            # Load the first .h5 file found
            model_path = model_files[0]
            new_model = load_model(model_path)
            #new_model = tf.keras.saving.load_model(model_path)
            print(f"Model loaded from: {model_path}")
        else:
            print("No .h5 model files found in the storage directory.")

        # 2. new_np
        # Loading the new_np object
        with open(os.path.join('storage', 'np_auto_labeling.pkl'), 'rb') as file:
            new_np = pickle.load(file)

        # 3. df
        # 4. labelList
        # 5. cols
        orig_df = pd.read_csv('storage/auto_label_df.csv')
        df = pd.read_csv('storage/auto_label_df.csv')

        # Create a temporary column for the rounded datetimes
        df['temp_datetime'] = pd.to_datetime(df['datetime']).dt.round('s')
        labelListDF = pd.read_csv('storage/label_list.csv')
        labelList = list(labelListDF)
        
        #cols = list(pd.read_csv('assets/feature_cols.csv'))
        # json change
        # Path to your JSON configuration
        config_path = os.path.join('', 'config.json')

        # Load the configuration
        with open(config_path, 'r') as file:
            config = json.load(file)

        # Extract lists from the configuration
        valid_features = config["valid_features"]
        features_to_omit = config["features_to_omit"]
        cols = config["features_to_plot"]
        conf_thresh = config["conf_thresh"]
        offset = config["offset_pred"]
        video_path = "assets/autolabel_video.mp4"

        # video's start time for calculating offsets
        # Adjusting the start time by the offset
        initial_time = pd.to_datetime(df['datetime'].iloc[0])
        video_start_time = initial_time + timedelta(seconds=int(offset))

        #6. window size
        window_size = config["window-and-step-size"]
        step = config["window-and-step-size"]

        timestamps = pd.to_datetime(df['datetime'])
        windows = [(timestamps[i], timestamps[min(i + window_size - 1, len(timestamps) - 1)]) for i in range(0, len(timestamps), window_size) if i + window_size <= len(timestamps)]
        testModel = model.CNN()
        testModel.setModel(new_model)

        modified_indices = {}

        #new dictionary for altering labels
        label_num_dict = {new_np.mapping[k] : k for k in new_np.mapping}
        print(label_num_dict)

        colorList = ('#4363d8', '#e6194b', '#3cb44b', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                    '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                    '#000000')
        colorDict = {label: (colorList[i % len(colorList)] if label != 'Undefined' else '#000000') 
                    for i, label in enumerate(labelListDF)}

        # Create a color map: For each activity code, map it to a specific color. 
        # Create a base color map for all labels except "Undefined"
        base_color_map = {label: colorList[i % len(colorList)] for i, label in enumerate(label_num_dict.keys()) if label != "Other"}

        # Explicitly set the color for "Other" to black
        base_color_map["Other"] = '#000000'

        # Invert the label_num_dict to map from numeric codes to string labels
        num_to_label_dict = {v: k for k, v in label_num_dict.items()}

        load_figure_template("bootstrap")
               
        npInput = new_np.y
        # Map npInput numeric codes to string labels
        npInput_labels = np.array([num_to_label_dict[code] for code in npInput])
        color_discrete_map = {label: base_color_map.get(label, '#000000') for label in np.unique(npInput_labels)}

        testModel.only_test_data(new_np.x, new_np.y)
        predictions = testModel.model.predict(new_np.x, verbose=0, batch_size = 32)

        lineGraph = go.Figure()
        plotly_umap = px.scatter()
        graph1 = px.scatter()
        graph2 = px.scatter()
        graph3 = px.scatter()

        embedding_df, embedding, umap_to_ts_mapping, predicted_labels = rerender_umap_and_update_df(predictions, df, windows, num_to_label_dict, base_color_map)

        flag_low_confidence_windows(df, predictions, windows, conf_thresh, modified_indices=modified_indices)
        update_dataframe_with_predictions(df, predictions, umap_to_ts_mapping, num_to_label_dict)

        # initialize plots
        plotly_umap = px.scatter(
            embedding_df, 
            x='x', 
            y='y', 
            color=npInput_labels,
            hover_name=npInput_labels,
            color_discrete_map=color_discrete_map,
            custom_data=['index']
        )
        plotly_umap.update_layout(
            autosize=False,
            legend_title_text="Class",
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(scaleanchor='x',scaleratio=1,),
            xaxis_title=None, yaxis_title=None,
        )

        # Create a new figure
        lineGraph = create_time_series_figure(valid_features, cols, df)

        update_timeseries_plot_with_flags(df, lineGraph)
        update_umap_plot_with_flags(embedding_df, plotly_umap)

        # WHAT NEEDS TO BE STORED
        # df, embedding_df, predictions, npInput_labels, npInput

        return create_main_layout(conf_thresh, labelList, plotly_umap, lineGraph, graph1, graph2, graph3, video_path)
    else:
        return html.Div()  # Minimal layout when not properly requested

###########
# UMAP 
###########
def rerender_umap_and_update_df(predictions, df, windows, num_to_label_dict, base_color_map, n_neighbors_value=15, min_dist_value=0.1):    
    npInput = np.argmax(predictions, axis=1)

    # Recreate the UMAP reducer and transform new predictions
    reducer = umap.UMAP(n_neighbors=n_neighbors_value, n_components=2, min_dist=min_dist_value)
    embedding = reducer.fit_transform(predictions)

    # Update the embedding DataFrame
    embedding_df = pd.DataFrame(embedding, columns=['x', 'y'])
    embedding_df['index'] = np.arange(len(embedding))

    # Map predictions to labels
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_labels = np.array([num_to_label_dict[code] for code in predicted_class_indices])
    color_discrete_map = {label: base_color_map.get(label, '#000000') for label in np.unique(predicted_labels)}
    embedding_df['label'] = predicted_labels

    # Recreate the window mapping
    umap_to_ts_mapping = {index: window for index, window in enumerate(windows)}
    embedding_df['window_start'] = embedding_df['index'].apply(lambda idx: umap_to_ts_mapping[idx][0] if idx in umap_to_ts_mapping else None)

    # Update labels and window_ids in original dataframe based on new mappings
    df['PredictedLabel'] = np.nan  # Reset previous labels
    df['window_id'] = -1  # Reset previous window IDs
    for index, row in embedding_df.iterrows():
        window_start, window_end = umap_to_ts_mapping[row['index']]
        mask = df['temp_datetime'].between(window_start, window_end)
        df.loc[mask, 'PredictedLabel'] = row['label']
        df.loc[mask, 'window_id'] = row['index']

    return embedding_df, embedding, umap_to_ts_mapping, predicted_labels

#################################################################################################################################################
# Confidence Functions (flagging/unflagging)
#################################################################################################################################################
def flag_low_confidence_windows(df, predictions, windows, conf_thresh=0.9, modified_indices=None):
    """
    Adds a 'flagged' column to the dataframe where each row within a window is flagged as True if the 
    highest prediction confidence for that window is below the specified threshold.

    Parameters:
    - df (pd.DataFrame): The DataFrame to which the 'flagged' column will be added.
    - predictions (np.array): A numpy array of prediction confidences from the model, corresponding to windows.
    - windows (list of tuples): A list where each tuple contains the start and end timestamps of a window.
    - conf_thresh (float): The confidence threshold, below which predictions are considered low confidence.

    Returns:
    - pd.DataFrame: The updated DataFrame with a new 'flagged' column.
    """

    if modified_indices:
        # Update predictions based on user feedback
        for index, new_confidence in modified_indices.items():
            predictions[index] = new_confidence

    if 'flagged' not in df.columns:
        df['flagged'] = False  # This ensures the column exists

    # First check if the lengths match
    if len(predictions) != len(windows):
        raise ValueError(f"Number of predictions ({len(predictions)}) does not match number of windows ({len(windows)})")
    
    # Compute the maximum prediction confidence for each window
    max_confidences = np.max(predictions, axis=1)
    # Create a boolean array where True indicates the max confidence is below the threshold
    low_confidence_flags = max_confidences < conf_thresh
    
    # Ensure datetime is in pandas datetime format for comparison
    df['temp_datetime'] = pd.to_datetime(df['datetime']).dt.round('s')
    
    # Add 'flagged' to Time-Series data
    for idx, (start_time, end_time) in enumerate(windows):
        if idx < len(low_confidence_flags) and low_confidence_flags[idx]:
            df.loc[df['temp_datetime'].between(start_time, end_time), 'flagged'] = True
        else:
            df.loc[df['temp_datetime'].between(start_time, end_time), 'flagged'] = False

    # Add 'flagged' to UMAP data
    embedding_df['flagged'] = [low_confidence_flags[idx] if idx < len(low_confidence_flags) else False for idx in range(len(embedding_df))]

def get_flagged_intervals(df):
    """
    Identifies intervals of consecutive True values in the 'flagged' column and calculates their start and end times based on the index and a specified window size.

    Parameters:
    - df (pd.DataFrame): DataFrame with a 'flagged' column and a datetime column named 'temp_datetime'.

    Returns:
    - List of tuples, where each tuple contains the start and end times of a flagged interval.
    """
    # Initialize 'flag_change' column to detect changes
    df['flag_change'] = df['flagged'].astype(int).diff().fillna(0).abs()
    
    # List to store start and end indices of changes
    change_indices = df[df['flag_change'] == 1].index.tolist()

    # Handle case when all values are flagged
    if df['flagged'].all():
        change_indices = [0]  # Start from the first index
        if df.iloc[-1]['flagged']:
            change_indices.append(len(df))  # Add the last index

    # Handle general case
    else:
        if df.iloc[-1]['flagged']:
            change_indices.append(len(df))  # Ensure the last segment is included if flagged

    # Calculate intervals based on detected changes
    intervals = []
    for start_idx, end_idx in zip(change_indices, change_indices[1:]):
        if df.loc[start_idx, 'flagged']:
            start_time = df.iloc[start_idx]['temp_datetime']
            end_time = df.iloc[end_idx - 1]['temp_datetime']
            intervals.append((start_time, end_time))

    return intervals

def update_umap_plot_with_flags(embedding_df, fig):
    flagged_points = embedding_df[embedding_df['flagged']]

    # Remove existing "Flagged Low Confidence" traces
    fig.data = [trace for trace in fig.data if trace.name != "Flagged Low Confidence"]
    
    if not flagged_points.empty:
        fig.add_trace(go.Scatter(
            x=flagged_points['x'],
            y=flagged_points['y'],
            mode='markers',
            marker=dict(
                symbol='x-open',
                color='black',
                size=12,
                line=dict(
                    color='black',  # High contrast solid border
                    width=2
                )
            ),
            name='Flagged Low Confidence',
            visible='legendonly'  # Set visibility to 'legendonly'
        ))

def update_timeseries_plot_with_flags(df, fig):
    """
    Adds shaded regions to a Plotly figure to indicate flagged intervals for low confidence in the data.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the time series data.
    - fig (plotly.graph_objs.Figure): The Plotly figure object to which the shaded regions will be added.
    - intervals (list of tuples): A list of tuples, each containing the start and end datetimes of a flagged interval.
    """
    
    # First, remove any existing "Flagged Low Confidence" traces or rectangles
    fig.data = [trace for trace in fig.data if trace.name != "Low Confidence"]
    fig.layout.shapes = [shape for shape in fig.layout.shapes if shape.name != "Low Confidence"]

    # Calculate intervals of low confidence
    intervals = get_flagged_intervals(df)

    # Check if there are any intervals to flag
    if intervals:
        # Add a 'dummy' trace for the legend toggle using a custom SVG path for marker
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                symbol="square",  # Standard square shape
                size=10,
                color='yellow',  # Transparent fill
                opacity= 0.5,
                line=dict(color='black', width=2),  # Black border
            ),
            legendgroup='low_confidence',
            name='Low Confidence',
            showlegend=True
        ))

        # confidence highlight on raw ts
        for start_time, end_time in intervals:
            fig.add_vrect(
                x0=start_time, x1=end_time,
                fillcolor="yellow", opacity=0.5,
                layer="below", annotation_text="Low Confidence", 
                annotation_position="top left", annotation_yshift=20,
            )

def update_dataframe_with_predictions(df, predictions, umap_to_ts_mapping, num_to_label_dict):
    """
    Update the DataFrame with predicted labels and confidence values based on model predictions.

    Parameters:
    - df (pd.DataFrame): The DataFrame to update.
    - predictions (np.ndarray): The softmax outputs from the model, assuming last axis sums to 1.
    - umap_to_ts_mapping (dict): A mapping from UMAP indices to time series data window indices.
    - num_to_label_dict (dict): A dictionary mapping from numeric indices to label names.
    """
    # Extract class indices and confidence values from predictions
    predicted_class_indices = np.argmax(predictions, axis=1)
    confidence_values = np.max(predictions, axis=1)

    # Check if 'confidence' column exists in DataFrame; if not, initialize it
    if 'confidence' not in df.columns:
        df['confidence'] = np.nan

    # Update the DataFrame with predictions and confidences
    for umap_index, (label_index, conf_value) in enumerate(zip(predicted_class_indices, confidence_values)):
        window_start, window_end = umap_to_ts_mapping[umap_index]
        mask = (df['temp_datetime'] >= window_start) & (df['temp_datetime'] <= window_end)
        df.loc[mask, 'PredictedLabel'] = num_to_label_dict[label_index]
        df.loc[mask, 'confidence'] = conf_value

##########################################################################################################

##########################################################################################################
# Raw Time Series Plot Creation
##########################################################################################################
# Retrieve start/end indices of all labels
def calculate_label_indices(df):
    tempDf = df.ne(df.shift())
    labelsStartIndex, labelsEndIndex = [], []
    index = -1
    for element in tempDf['label']:
        index += 1
        if element == True:
            labelsStartIndex.append(index)
            if labelsStartIndex and index > 0:
                labelsEndIndex.append(index - 1)
    labelsEndIndex.append(len(tempDf['label']) - 1)
    return labelsStartIndex, labelsEndIndex

def create_time_series_figure(valid_features, cols, df, additional_shapes=None):
    # Create a new figure
    fig = go.Figure()
    labelsStartIndex, labelsEndIndex = calculate_label_indices(df)

    # Define the hover template
    hovertemplate = (
        "Datetime: %{x}<br>"
        "Value: %{y}<br>"
        "Label: %{customdata[0]}<br>"
        "Confidence: %{customdata[1]}<extra></extra>"
    )
    fig.update_layout(hovermode='x unified')

    # Add each valid feature as a trace with hover information
    for feature in valid_features:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df[feature],
                name=feature,
                visible="legendonly" if feature not in cols else True,
                customdata=df[['label', 'confidence']],
                hovertemplate=hovertemplate
            )
        )

    # Add rectangles for each label
    for start_idx, end_idx in zip(labelsStartIndex, labelsEndIndex):
        start_date, end_date = df['datetime'].iloc[start_idx], df['datetime'].iloc[end_idx]
        label = df['label'].iloc[start_idx]
        confidence = df['confidence'].iloc[start_idx]

        # Choose color, defaulting to black if label not found in colorDict
        color = colorDict.get(label, '#000000')

        # Add a rectangle shape for each labeled interval
        fig.add_shape(
            type="rect",
            x0=start_date, y0=0, x1=end_date, y1=1,
            xref="x", yref="paper",
            fillcolor=color, opacity=0.3, layer="below", line_width=0.5
        )

        if label != "Undefined":
            midpoint_index = start_idx + (end_idx - start_idx) // 2
            midpoint_date = df['datetime'].iloc[midpoint_index]
            fig.add_annotation(
                x=midpoint_date, y=1.0,  # Top of the plot area
                text=f"{label}",
                showarrow=False,
                yref="paper",  # Use 'paper' reference for y to align with the top of the plot area
                yanchor="top",  # Anchor the text at the top
                font=dict(color='#000000'),
                yshift=3  # Shift up by 5 units to keep the text inside the plot area
            )

    # Add dummy scatter traces for legend entries
    unique_labels = df['label'].dropna().unique()
    for label in unique_labels:
        color = colorDict.get(label, '#000000')  # Get color from color dictionary
        fig.add_trace(go.Scatter(
            x=[None],  # No actual data points
            y=[None],
            mode='markers',
            marker=dict(
                color=color,
                symbol='square',  # Use 'square' to represent the marker as a square
                size=10,  # Adjust size to make it visible and appropriately sized in the legend
                opacity=0.5
            ),
            name=label
        ))

    if additional_shapes:
        for shape in additional_shapes:
            fig.add_shape(shape)

    # Update layout
    fig.update_layout(
        xaxis=dict(
            title="Datetime",
            rangeslider=dict(visible=True),
            type="date"
        ),
        legend=dict(title=dict(text="Features")),
        clickmode='event+select'
    )

    return fig

###########################################################################################
# UMAP
###########################################################################################
def create_umap_figure(embedding_df, npInput_labels):
    npInput_labels = np.array([num_to_label_dict[code] for code in npInput])
    color_discrete_map = {label: base_color_map.get(label, '#000000') for label in np.unique(npInput_labels)}
    umap_fig = px.scatter(
        embedding_df, x='x', y='y', color=npInput_labels, hover_name=npInput_labels,
        color_discrete_map=color_discrete_map, custom_data=['index']
    ).update_layout(autosize=False, xaxis_title=None, yaxis_title=None, legend_title_text="Class")
    return umap_fig

def highlight_umap_point(umap_fig, selected_index, nearest_overall_index, nearest_same_type_index):
    # Define marker styles based on the source
    selected_marker = {
        'color': 'black', 'size': 10, 'symbol': 'circle-open',
        'line': {'color': 'black', 'width': 2}
    }
    nearest_overall_marker = {
        'color': 'red', 'size': 8, 'symbol': 'diamond',
        'line': {'color': 'black', 'width': 2}
    }
    nearest_same_type_marker = {
        'color': 'purple', 'size': 10, 'symbol': 'star',
        'line': {'color': 'black', 'width': 2}
    }

    # Highlight selected point
    selected_point = embedding_df.iloc[selected_index]
    umap_fig.add_trace(go.Scatter(
        x=[selected_point['x']],
        y=[selected_point['y']],
        mode='markers',
        marker = selected_marker,
        name=f'Selected Point',
        showlegend=True
    ))

    # Highlight nearest overall neighbor
    if nearest_overall_index is not None:
        overall_neighbor = embedding_df.iloc[nearest_overall_index]
        umap_fig.add_trace(go.Scatter(
            x=[overall_neighbor['x']],
            y=[overall_neighbor['y']],
            mode='markers',
            marker = nearest_overall_marker,
            name=f'Nearest Neighbor: Overall',
            showlegend=True
        ))

    # Highlight nearest same type neighbor
    if nearest_same_type_index is not None and nearest_same_type_index != nearest_overall_index:
        same_type_neighbor = embedding_df.iloc[nearest_same_type_index]
        umap_fig.add_trace(go.Scatter(
            x=[same_type_neighbor['x']],
            y=[same_type_neighbor['y']],
            mode='markers',
            marker = nearest_same_type_marker,
            name=f'Nearest Neighbor: Same Type',
            showlegend=True
        ))

    return umap_fig

def nearestNeighbor(embedding, pointIndex, y):
    min_o =  math.inf
    min_c = math.inf
    nearest_neighbor = None
    point_loc = embedding[pointIndex]
    n_color = pointIndex
    for i in range(len(embedding)):
        distance = pow(pow(point_loc[0] - embedding[i][0], 2) + pow(point_loc[1] - embedding[i][1] , 2), 0.5)
        if distance < min_c and distance != 0 and y[i] == y[pointIndex]:
            n_color = i
            min_c = distance
        if distance < min_o and distance != 0:
            n_overall = i
            min_o = distance
    return n_overall, n_color
    
def highlight_time_series_window(plot_fig, selected_id, df, new_np, embedding, npInput):
    selected_style = {'fillcolor': "grey", 'opacity': 0.5, 'line': {'color': 'white', 'width': 2}}
    nearest_overall_style = {'fillcolor': "red", 'opacity': 0.5, 'line': {'color': 'white', 'width': 2}}
    nearest_same_type_style = {'fillcolor': "purple", 'opacity': 0.5, 'line': {'color': 'white', 'width': 2}}
    marker_style = lambda color: {'symbol': "square", 'color': color, 'size': 10, 'line': {'color': color, 'width': 10}}

    # Highlight the selected point's time window & add dummy trace for legend
    start, end = new_np.time[selected_id]
    plot_fig.add_vrect(x0=start, x1=end, layer="below", **selected_style)
    plot_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                    marker=marker_style(selected_style['fillcolor']),
                                    name='Selected Point'))
    
    # Find and highlight nearest neighbors
    near_o, near_c = nearestNeighbor(embedding, selected_id, npInput)

    # Highlight the nearest neighbor overall's time window & add dummy trace for legend
    if near_o is not None:
        start, end = new_np.time[near_o]
        plot_fig.add_vrect(x0=start, x1=end, layer="below", **nearest_overall_style)
        plot_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                    marker=marker_style(nearest_overall_style['fillcolor']),
                                    name='Nearest Neighbor: Overall'))


    # Highlight the nearest neighbor of the same type's time window & add dummy trace for legend
    if near_c is not None and near_c != near_o:
        start, end = new_np.time[near_c]
        plot_fig.add_vrect(x0=start, x1=end, layer="below", **nearest_same_type_style)
        plot_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                    marker=marker_style(nearest_same_type_style['fillcolor']),
                                    name='Nearest Neighbor: Same Type'))

def find_umap_point_from_ts(clicked_datetime, windows):
    clicked_datetime = pd.to_datetime(clicked_datetime)
    # Iterate over each window to find where the clicked_datetime falls
    for umap_index, (start, end) in enumerate(windows):
        if start <= clicked_datetime <= end:
            return umap_index
    return None

def update_label(data_index, new_label, npInput, npInput_labels, label_num_dict, manual_confidence=True):
    # update umap label info 
    if data_index is not None and new_label in label_num_dict:
        # Update only the label for the specific data point
        new_label_code = label_num_dict[new_label]
        npInput[data_index] = new_label_code
        npInput_labels[data_index] = new_label

        # Update confidence to the maximum if manual confidence setting is enabled
        if manual_confidence:
            modified_indices[data_index] = 1.0
            predictions[data_index, :] = 0  # Reset predictions for this index
            predictions[data_index, new_label_code] = 1.0  # Set the chosen label's confidence to 1
        
    # Update predictions and confidences in the main DataFrame
    if modified_indices:
        for idx, conf in modified_indices.items():
            # Update predictions if necessary
            window_start, window_end = umap_to_ts_mapping[idx]
            mask = (df['temp_datetime'] >= window_start) & (df['temp_datetime'] <= window_end)
            df.loc[mask, 'PredictedLabel'] = new_label
            df.loc[mask, 'label'] = new_label
            df.loc[mask, 'confidence'] = conf

            # Also update embedding_df if it contains a corresponding 'index' column
            if 'index' in embedding_df.columns:
                embedding_mask = (embedding_df['index'] == idx)
                embedding_df.loc[embedding_mask, 'label'] = new_label
                embedding_df.loc[embedding_mask, 'confidence'] = conf

    update_dataframe_with_predictions(df, predictions, umap_to_ts_mapping, num_to_label_dict)
    return modified_indices

def update_subgraphs(graph1, graph2, graph3, id, df, new_np, embedding, npInput):
    # Ensure 'datetime' column is in proper datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract indices for the nearest neighbors once to avoid recomputation
    nearest_overall_index, nearest_same_type_index = nearestNeighbor(embedding, id, npInput)

    # Prepare titles for clarity in the graphs
    titles = ["Selected Point", "Nearest Neighbor: Overall", "Nearest Neighbor: Same Type"]
    
    # Check if the nearest overall and same type are the same
    if nearest_overall_index == nearest_same_type_index:
        titles[2] = "Same as Nearest Overall"
        graph3.data = []  # Optionally clear any existing data
        graph3.update_layout(
            title=titles[2],
            showlegend=True,
            xaxis_title="Datetime",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            annotations=[{
                'text': "identical to nearest overall",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 16}
            }]
        )

    # Iterate over each graph object along with its corresponding data index and title
    for g, idx, title in zip([graph1, graph2, graph3], [id, nearest_overall_index, nearest_same_type_index], titles):
        if idx is not None and title != "Same as Nearest Overall":
            # Ensure timestamps are in datetime format
            start_time = pd.to_datetime(str(new_np.time[idx][0]))
            end_time = pd.to_datetime(str(new_np.time[idx][1]))

            # Extract the subset of dataframe for the given index
            g_data = df.loc[df["datetime"].between(start_time, end_time)]
            g.layout.annotations = []  # Important to remove old annotations
            g.data = []  # Clear existing data

            # Populate the graph with new data
            for feature in valid_features:
                g.add_trace(
                    go.Scatter(
                        x=g_data["datetime"],
                        y=g_data[feature],
                        name=feature,
                        visible="legendonly" if feature not in cols else True
                    )
                )

            # Update layout with the appropriate title
            g.update_layout(title=title, showlegend=True, xaxis_title="Datetime", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        elif title == "Same as Nearest Overall":  # Ensure this graph remains blank
            g.update_layout(showlegend=False)
        else:  # Handle no data case for other graphs
            g.data = []
            g.update_layout(title=f"No Data for {title}")

    
def clear_highlights(fig, source):
    highlight_trace_names = ['Selected Point', 'Nearest Neighbor: Overall', 'Nearest Neighbor: Same Type']

    # Collect non-highlight traces
    new_data = [trace for trace in fig.data if trace.name not in highlight_trace_names]
    
    # Construct a new figure with the non-highlight traces and a reset layout
    if source == "umap":
        cleaned_fig = go.Figure(data=new_data)
        cleaned_fig.update_layout(
            autosize=fig.layout.autosize,
            # Correctly access and set the legend title
            legend_title_text=fig.layout.legend.title.text if fig.layout.legend and fig.layout.legend.title else None,
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(scaleanchor='x', scaleratio=1),
            xaxis_title=None,
            yaxis_title=None,
        )
        # Explicitly clear all shapes for UMAP
        cleaned_fig.layout.shapes = []
    elif source == "ts":
        cleaned_fig = go.Figure(data=new_data)
        cleaned_fig.update_layout(
            xaxis_title="Datetime",
            # Correctly access and set the legend title
            legend=dict(title=dict(text=fig.layout.legend.title.text if fig.layout.legend and fig.layout.legend.title else "Features")),
            clickmode='event+select'
        )
        # Clear shapes such as vrects and annotations
        cleaned_fig.layout.shapes = []
        cleaned_fig.layout.annotations = []

    return cleaned_fig

def video_time_to_window_index(video_time_seconds, video_start_time, windows):
    # Convert video time in seconds to the corresponding window index in the data.

    # Calculate the datetime corresponding to the current video time
    current_datetime = video_start_time + pd.to_timedelta(video_time_seconds, unit='s')
    
    # Find the window index where this datetime falls
    for index, (start_time, end_time) in enumerate(windows):
        if start_time <= current_datetime <= end_time:
            return index
    return None  # Return None if no window is found (unlikely but safe to handle)

def update_dataframe(df, start, end, label, confidence):
    # Ensure the 'datetime' column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Convert start and end times to datetime
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # Find the closest start and end indices in the DataFrame
    # If exact match is not found, it finds the nearest date
    startIndex = df.index[df['datetime'] >= start][0]
    endIndex = df.index[df['datetime'] <= end][-1]

    # Update the label only if provided
    if label is not None:
        df.loc[startIndex:endIndex, 'label'] = label

    # Update the confidence only if provided
    if confidence is not None:
        df.loc[startIndex:endIndex, 'confidence'] = confidence
    create_time_series_figure(valid_features, cols, df)
###########################################################################################

##########################################################################################################
# Callbacks
##########################################################################################################
@callback(
    [
        Output('umap-graph', 'figure'),
        Output('plot-clicked', 'figure'),
        Output('graph1', 'figure'),
        Output('graph2', 'figure'),
        Output('graph3', 'figure'),
        Output('store_data', 'data'),
        Output('umap-video-player', 'seekTo'),
    ],
    [
        Input('umap-graph', 'clickData'),
        Input('plot-clicked', 'clickData'),
        Input('graph1', 'clickData'),
        Input('graph2', 'clickData'),
        Input('graph3', 'clickData'),
        Input('button', 'n_clicks'),
        Input('submit_button', 'n_clicks'), # umap params
        Input('vid_sync_button', 'n_clicks'),
        Input('btn-manual-label', 'n_clicks'), # added input for manual label
        Input('plot-clicked', 'relayoutData') # added input for relayoutData
    ],
    [
        State("dropdown", "value"),
        State("store_data", "data"),
        State('n_neighbors_input', 'value'), # umap params
        State('min_dist_slider', 'value'), # umap params
        State('umap-video-player', 'currentTime'),
        State('ts-fill-ui-checkbox', 'value'), # added state for UI checkbox
        State('ts-start-input', 'value'), # added state for start input
        State('ts-end-input', 'value'), # added state for end input
        State('ts-label-selection', 'value'), # added state for label selection
        State('ts-confidence-selection', 'value') # added state for confidence selection
    ] 
)
def update_app(umap_clickData, plot_clickData, graph1_clickData, graph2_clickData, graph3_clickData, label_n_clicks, umap_n_clicks, vid_sync_n_clicks, ts_label_n_clicks, ts_relayout_data, value, data, n_neighbors, min_dist, current_time, fill_ui_value, start_input, end_input, label_selection, confidence_selection):
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]
    # intialize video seek time
    video_seek_time = dash.no_update

    # Initialize figures with current state
    umap_fig = create_umap_figure(embedding_df, npInput_labels)
    update_umap_plot_with_flags(embedding_df, umap_fig)
    plot_fig = create_time_series_figure(valid_features, cols, df)
    update_timeseries_plot_with_flags(df, plot_fig)
    umap_fig.update_layout(autosize=False, xaxis_title=None, yaxis_title=None, legend_title_text="Class")
    graph1 = px.scatter()
    graph2 = px.scatter()
    graph3 = px.scatter()

    # TO DO: BUG FIX
    # reset params
    if triggered == 'submit_button':
        rerender_umap_and_update_df(predictions, df, windows, num_to_label_dict, base_color_map, n_neighbors_value=n_neighbors, min_dist_value=min_dist)
        flag_low_confidence_windows(df, predictions, windows, conf_thresh=conf_thresh, modified_indices=None)
        update_umap_plot_with_flags(embedding_df, umap_fig)
        update_timeseries_plot_with_flags(df, plot_fig)
        return [create_umap_figure(embedding_df, predicted_labels)] + [dash.no_update]*6
    
    # vid to data sync button click
    if triggered == 'vid_sync_button':
        umap_fig = clear_highlights(umap_fig, "umap")
        plot_fig = clear_highlights(plot_fig, "ts")
        # ensure flagged labels are still shown
        update_umap_plot_with_flags(embedding_df, umap_fig)
        update_timeseries_plot_with_flags(df, plot_fig)

        # Use the helper function to find the corresponding window index
        window_index = video_time_to_window_index(current_time, video_start_time, windows)
        
        # If a valid window index is found, proceed to highlight and update plots
        if window_index is not None:
            # Assuming that the UMAP and window mappings are properly set up:
            selected_index = window_index  # This might need adjustment based on your specific setup
            nearest_overall_index, nearest_same_type_index = nearestNeighbor(embedding, selected_index, npInput)

            umap_fig = highlight_umap_point(umap_fig, selected_index, nearest_overall_index, nearest_same_type_index)
            highlight_time_series_window(plot_fig, selected_index, df, new_np, embedding, npInput)
            update_subgraphs(graph1, graph2, graph3, selected_index, df, new_np, embedding, npInput)
        else:
            print(f"No corresponding window found for video time {current_time}s")
        
        video_seek_time = dash.no_update
        
    # Click on time-series plots
    if triggered in ['plot-clicked', 'graph1', 'graph2', 'graph3']:
        if triggered == 'plot-clicked':
            clickData = plot_clickData
        elif triggered == 'graph1':
            clickData = graph1_clickData
        elif triggered == 'graph2':
            clickData = graph2_clickData
        elif triggered == 'graph3':
            clickData = graph3_clickData
        else:
            raise PreventUpdate 
        if clickData is not None:
            # TO DO: change to call create umap
            clicked_datetime = pd.to_datetime(clickData['points'][0]['x'])
            npInput_labels_updated = np.array([num_to_label_dict[code] for code in npInput])
            color_discrete_map_updated = {label: base_color_map.get(label, '#000000') for label in np.unique(npInput_labels_updated)}
            umap_fig = px.scatter(
                embedding_df, x='x', y='y', color=npInput_labels_updated, hover_name=npInput_labels_updated,
                color_discrete_map=color_discrete_map_updated, custom_data=['index']
            ).update_layout(autosize=False, xaxis_title=None, yaxis_title=None, legend_title_text="Class")
            umap_point_index = find_umap_point_from_ts(clicked_datetime, windows)
            nearest_overall_index, nearest_same_type_index = nearestNeighbor(embedding, umap_point_index, npInput)
        else:
            return dash.no_update

        if umap_point_index is not None:
            window_start, window_end = umap_to_ts_mapping.get(umap_point_index, (None, None))
            if window_start and window_end:
                umap_fig = highlight_umap_point(umap_fig, umap_point_index, nearest_overall_index, nearest_same_type_index)
                highlight_time_series_window(plot_fig, umap_point_index, df, new_np, embedding, npInput)
                update_subgraphs(graph1, graph2, graph3, umap_point_index, df, new_np, embedding, npInput)
            
            # Calculate video seek time based on the time window start
            if window_start:
                video_seek_time = (window_start - video_start_time).total_seconds()

    #if triggered == 'plot-clicked' and 'relayoutData' in ctx.triggered[0]['prop_id']:
    if 'relayoutData' in ctx.triggered[0]['prop_id']:
        # Check if the fill-ui-checkbox is checked and there is a selected range
        if 'fill-ui' in fill_ui_value and 'xaxis.range[0]' in ts_relayout_data and 'xaxis.range[1]' in ts_relayout_data:
            start_date = ts_relayout_data['xaxis.range[0]']
            end_date = ts_relayout_data['xaxis.range[1]']
            # Add the shape for the highlighted region
            highlight_shape = {
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': start_date,
                'x1': end_date,
                'y0': 0,
                'y1': 1,
                'fillcolor': 'rgba(0, 0, 255, 0.2)',
                'line': {'width': 0},
            }
            additional_shapes=[highlight_shape]
            plot_fig = create_time_series_figure(valid_features, cols, df, additional_shapes)

    # click on umap
    if triggered == 'umap-graph':
        umap_fig = clear_highlights(umap_fig, "umap")
        npInput_labels_updated = np.array([num_to_label_dict[code] for code in npInput])
        color_discrete_map_updated = {label: base_color_map.get(label, '#000000') for label in np.unique(npInput_labels_updated)}
        umap_fig = px.scatter(
            embedding_df, x='x', y='y', color=npInput_labels_updated, hover_name=npInput_labels_updated,
            color_discrete_map=color_discrete_map_updated, custom_data=['index']
        ).update_layout(autosize=False, xaxis_title=None, yaxis_title=None, legend_title_text="Class")
        #plot_fig = clear_highlights(plot_fig, "ts")
        # ensure flagged labels are still shown
        update_umap_plot_with_flags(embedding_df, umap_fig)
        update_timeseries_plot_with_flags(df, plot_fig)

        # Extract the index of the clicked point in the UMAP plot
        selected_index = umap_clickData["points"][0]["customdata"][0]
        nearest_overall_index, nearest_same_type_index = nearestNeighbor(embedding, selected_index, npInput)

        # Use the mapping to find the corresponding time series window
        window_start, window_end = umap_to_ts_mapping.get(selected_index, (None, None))
        if window_start and window_end:
            umap_fig = highlight_umap_point(umap_fig, selected_index, nearest_overall_index, nearest_same_type_index)
            highlight_time_series_window(plot_fig, selected_index, df, new_np, embedding, npInput)
            update_subgraphs(graph1, graph2, graph3, selected_index, df, new_np, embedding, npInput)

        window_start = embedding_df.loc[selected_index, 'window_start']
        # If window_start is a datetime object, convert it to the video timeline
        if not pd.isnull(window_start):
            # Ensure window_start is a datetime object
            if isinstance(window_start, str):
                window_start = datetime.strptime(window_start, '%Y-%m-%d %H:%M:%S')
            
            # Calculate video seek time in seconds
            video_seek_time = (window_start - video_start_time).total_seconds()

        else:
            raise PreventUpdate

    # label change on umap
    if "button" == ctx.triggered_id:
        selected_index = umap_clickData["points"][0]["customdata"][0] if umap_clickData else None
        modified_indices = update_label(selected_index, value, npInput, npInput_labels, label_num_dict, manual_confidence=True)
        # TO DO: change to call create umap
        npInput_labels_updated = np.array([num_to_label_dict[code] for code in npInput])
        color_discrete_map_updated = {label: base_color_map.get(label, '#000000') for label in np.unique(npInput_labels_updated)}
        umap_fig = px.scatter(
            embedding_df, x='x', y='y', color=npInput_labels_updated, hover_name=npInput_labels_updated,
            color_discrete_map=color_discrete_map_updated, custom_data=['index']
        ).update_layout(autosize=False, xaxis_title=None, yaxis_title=None, legend_title_text="Class")
        flag_low_confidence_windows(df, predictions, windows, conf_thresh=conf_thresh, modified_indices=modified_indices)
        update_umap_plot_with_flags(embedding_df, umap_fig)
        plot_fig = create_time_series_figure(valid_features, cols, df)
        update_timeseries_plot_with_flags(df, plot_fig)

    if triggered == 'btn-manual-label':
        # DF data
        # Update label and confidence data
        start = start_input
        end = end_input
        start_dt = pd.to_datetime(start_input)
        end_dt = pd.to_datetime(end_input)
        label = label_selection
        confidence = confidence_selection
        
        # Update DataFrame with new label and confidence data
        update_dataframe(df, start, end, label, confidence)

        # Recalculate label indices after updating DataFrame
        labelsStartIndex, labelsEndIndex = calculate_label_indices(df)

        # recreate
        plot_fig = create_time_series_figure(valid_features, cols, df)

        # Find which windows are affected by the time range selected
        affected_windows = []
        for i, window in enumerate(windows):
            if start_dt <= window[1] and end_dt >= window[0]:  # Check if the selected range overlaps with the window
                affected_windows.append(i)
        
        # Update predictions for affected windows
        if label in label_num_dict:
            new_label_code = label_num_dict[label]
            for idx in affected_windows:
                predictions[idx, :] = 0  # Reset the prediction
                predictions[idx, new_label_code] = 1.0  # Set the chosen label with full confidence


        rerender_umap_and_update_df(predictions, df, windows, num_to_label_dict, base_color_map, n_neighbors_value=n_neighbors, min_dist_value=min_dist)

        # Update the UMAP plot
        umap_fig = create_umap_figure(embedding_df, predicted_labels)
        flag_low_confidence_windows(df, predictions, windows, conf_thresh=conf_thresh, modified_indices=None)
        update_timeseries_plot_with_flags(df, plot_fig)
        update_umap_plot_with_flags(embedding_df, umap_fig)

    return umap_fig, plot_fig, graph1, graph2, graph3, data, video_seek_time

@callback(
    Output('save_status', 'children'),
    [Input('save_changes_button', 'n_clicks')],
    #State('store_data', 'data'),  # Ensure you have the current state of the data
    prevent_initial_call=True
)
def save_changes(n_clicks, data):
    if n_clicks > 0:
        try:
            # Saving the updated 'new_np' object # relevant for retraining
            # with open(os.path.join('labeled_data', 'time_series_processed_data.pkl'), 'wb') as file:
            #     pickle.dump(data['new_np'], file)  
            
            # Saving the DataFrame
            df_path = os.path.join('labeled_data', 'labeled_data.csv')  # Default save location
            # Columns to drop
            columns_to_drop = ['temp_datetime', 'PredictedLabel', 'window_id', 'flagged', 'confidence', 'flag_change']
            # Dropping the columns
            df.drop(columns=columns_to_drop, inplace=True)
            df.to_csv(df_path, index=False)

            return 'Changes and data saved successfully!'
        except Exception as e:
            return f'Error saving changes: {e}'
    return ''

# checkbox for auto fill start end datetime selection on manual graph callbacks
@callback(
    Output('ts-start-input', 'value'),
    Input('plot-clicked', 'relayoutData'),
    State('ts-fill-ui-checkbox', 'value')
)
def update_start_input(relayoutData, fill_ui_value):
    if relayoutData and 'xaxis.range[0]' in relayoutData:
        if 'fill-ui' in fill_ui_value:
            start_date = relayoutData['xaxis.range[0]']
            return start_date
    return None

@callback(
    Output('ts-end-input', 'value'),
    Input('plot-clicked', 'relayoutData'),
    State('ts-fill-ui-checkbox', 'value')
)
def update_end_input(relayoutData, fill_ui_value):
    if relayoutData and 'xaxis.range[1]' in relayoutData:
        if 'fill-ui' in fill_ui_value:
            end_date = relayoutData['xaxis.range[1]']
            return end_date
    return None
    