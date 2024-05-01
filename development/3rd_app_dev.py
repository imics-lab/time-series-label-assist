# Dash framework and core components
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_player
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash import callback_context
from dash.exceptions import PreventUpdate

# Dash Bootstrap Components for styling
import dash_bootstrap_components as dbc

# Plotly for interactive plotting
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Pandas for data manipulation
import pandas as pd
import numpy as np

# Additional Python libraries
import time
from datetime import timedelta, datetime

import sys
import os

from tyler_code import split, model
import umap
import math

import pickle
import glob
import tensorflow as tf
import json
#################################################################################################################################################
# FETCH
#########

predictions = np.load('assets/predictions.npy')
dfA = pd.read_csv('assets/auto_label_df.csv')
#dfA = pd.read_csv('assets/manual_label_df.csv')
video_path = 'assets/autolabel_video.mp4' # local video in asset directory (uploaded during provide dataset to autolabel)
labelListDF = pd.read_csv('assets/label_list.csv')
labelList = list(labelListDF)
# Path to your JSON configuration
working_dir = os.getcwd()
assets_dir = os.path.join(working_dir, "assets")
config_path = os.path.join(assets_dir, 'config.json')

# video's start time for calculating offsets (TO DO: FROM CONFIG)
video_start_time = datetime.strptime('2021-10-08 16:50:50', '%Y-%m-%d %H:%M:%S')

# Load the configuration
with open(config_path, 'r') as file:
    config = json.load(file)

# Extract lists from the configuration
valid_features = config["valid_features"]
features_to_omit = config["features_to_omit"]
cols = config["features_to_show"]
conf_thresh = config["conf_thresh"]

colorList = ('#4363d8', '#e6194b', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
             '#000000')
# This assigns a specific color from colorList to each label in provided label list ** IF LABEL IS UNDEFINED COLOR IS BLACK**
colorDict = {label: (colorList[i % len(colorList)] if label != 'Other' else '#000000') 
             for i, label in enumerate(labelListDF)}

# Confidence list & dictionary
confidenceValues = ("High", "Medium", "Low", "Undefined")
confidenceColors = ('#3cb44b', '#ffe119', '#FF0000', '#000000')

confDict = dict(zip(confidenceValues, confidenceColors))

size = len(dfA)
# Makes list and assigns highest value to each index
labelLine = []
for i in range(size):
    labelLine.append(1)

########
# UMAP #
########
# 1. model_select.value
# Path to the assets directory
working_dir = os.getcwd()
assets_dir = os.path.join(working_dir, "assets")

# Search for .h5 files in the assets directory
model_files = glob.glob(os.path.join(assets_dir, '*.h5'))

# Check if any .h5 files were found
if model_files:
    # Load the first .h5 file found
    model_path = model_files[0]
    #new_model = load_model(model_path)
    new_model = tf.keras.saving.load_model(model_path)
    print(f"Model loaded from: {model_path}")
else:
    print("No .h5 model files found in the assets directory.")

# 3. window & step size; 
# this got hard coded for some reason, explore later.
window_size = 96
step_size = 96

# dict of mapping for each label
class_index_to_label = {i: label for i, label in enumerate(labelList)}

# Find the index of the highest probability (predicted class index) in each prediction array
predicted_class_indices = [np.argmax(pred) for pred in predictions]

# Replicate the class indices according to the window and step size
expanded_class_indices = []

# Loop through each predicted class index
for i in range(len(predicted_class_indices)):
    # Extend the expanded_class_indices list with window_size copies of the current class index
    expanded_class_indices.extend([predicted_class_indices[i]] * window_size)

    # Adjust for overlap if not the last index and if there is an actual overlap
    if i + 1 < len(predicted_class_indices) and step_size < window_size:
        overlap = window_size - step_size
        # Only remove the overlap if it does not exceed the list's length
        if len(expanded_class_indices) > overlap:
            expanded_class_indices = expanded_class_indices[:-overlap]

# Handle case where expanded_class_indices is longer than the original data
if len(dfA) < len(expanded_class_indices):
    expanded_class_indices = expanded_class_indices[:len(dfA)]

# Handle remaining rows, if any
remaining_rows = len(dfA) - len(expanded_class_indices)
if remaining_rows > 0:
    expanded_class_indices.extend([predicted_class_indices[-1]] * remaining_rows)

# Convert these indices to label names using the provided mapping
label_predictions = [class_index_to_label.get(index, "Other") for index in expanded_class_indices]

dfA['label'] = label_predictions

if "pred_labels" in dfA.columns:
    dfA['label'] = dfA['pred_labels']
    dfA = dfA.drop('pred_labels', axis=1)

if dfA.index.name != "datetime":
    dfA = dfA.set_index('datetime')

np_pred = split.TimeSeriesNP(window_size, step_size)
np_pred.setArrays(dfA, encode = True, one_hot_encode=False ,labels =labelList)
npInput = np_pred.y

dfA = dfA.reset_index()

# create datastructures to hold where each prediction is
timestamps = pd.to_datetime(dfA['datetime'])  
#windows = [(timestamps[i], timestamps[min(i + window_size - 1, len(timestamps) - 1)]) for i in range(0, len(timestamps), window_size)]
windows = [(timestamps[i], timestamps[min(i + window_size - 1, len(timestamps) - 1)]) for i in range(window_size, len(timestamps), window_size)]
valid_start = windows[1][0]  # Start of the second window
valid_end = windows[-2][1]  # End of the second-to-last window

dfA['temp_datetime'] = pd.to_datetime(dfA['datetime']).dt.round('s')

# Trim the DataFrame to only include rows within the valid range
valid_df = dfA[(dfA['temp_datetime'] >= valid_start) & (dfA['temp_datetime'] <= valid_end)]

testModel = model.CNN()
testModel.setModel(new_model)

#new dictionary for altering labels
label_num_dict = {np_pred.mapping[k] : k for k in np_pred.mapping}

# Create a base color map for all labels except "Undefined"
base_color_map = {label: colorList[i % len(colorList)] for i, label in enumerate(label_num_dict.keys()) if label != "Other"}
base_color_map["Other"] = '#000000'

# Invert the label_num_dict to map from numeric codes to string labels
num_to_label_dict = {v: k for k, v in label_num_dict.items()}

# Map npInput numeric codes to string labels
npInput_labels = np.array([num_to_label_dict[code] for code in npInput])
color_discrete_map = {label: base_color_map.get(label, '#000000') for label in np.unique(npInput_labels)}

testModel.only_test_data(np_pred.x, np_pred.y)
#predictions = testModel.model.predict(np_pred.x, verbose=0, batch_size = 32)

reducer = umap.UMAP(n_neighbors = 15, n_components =2)
embedding = reducer.fit_transform(predictions)

embedding_df = pd.DataFrame(embedding, columns=['x', 'y'])
embedding_df['index'] = np.arange(len(embedding))

umap_to_ts_mapping = {umap_index: (window_start, window_end) for umap_index, (window_start, window_end) in enumerate(windows)}

dfA["confidence"] = "High"

##########################################################################################################
# Initialize the 'window_id' column with a default value (e.g., -1)
dfA['window_id'] = -1

# Iterate through the list of windows to assign window_id
for window_index, (start_time, end_time) in enumerate(windows):
    # Select rows where 'datetime' falls within the current window's range
    # And assign the current window index to the 'window_id' column
    dfA.loc[dfA['temp_datetime'].between(start_time, end_time), 'window_id'] = window_index

# Display the DataFrame to verify the 'window_id' assignment
#df.to_csv("windows-on-data.csv")

# Assign window_id to each row in df based on the timestamp windows
for window_index, (start_time, end_time) in enumerate(windows):
    mask = dfA['temp_datetime'].between(start_time, end_time)
    dfA.loc[mask, 'window_id'] = window_index
# Add a 'window_id' column to embedding_df
embedding_df['window_id'] = -1  # Initialize with a default value

# Iterate over each row in embedding_df to assign the correct window_id
for index, row in embedding_df.iterrows():
    umap_index = row['index']
    if umap_index in umap_to_ts_mapping:
        window_start, window_end = umap_to_ts_mapping[umap_index]
        # Find a row in df that falls within the window_start and window_end
        # and use its window_id for the umap_index
        sample_row = dfA[dfA['temp_datetime'].between(window_start, window_end)].head(1)
        if not sample_row.empty:
            window_id = sample_row['window_id'].values[0]
            embedding_df.at[index, 'window_id'] = window_id

embedding_df['window_start'] = embedding_df['index'].apply(lambda idx: umap_to_ts_mapping[idx][0] if idx in umap_to_ts_mapping else None) # get times for each umap point

#################################################################################################################################################
# confidence stuff (flagging)
#################################
# Initialize the 'flagged' column to 0 for all rows in dfA
dfA['flagged'] = 0

# Loop over each prediction and its corresponding window
for i, pred in enumerate(predictions):
    # Check if the maximum confidence value in the prediction is below the threshold
    flagged_value = 1 if max(pred) < conf_thresh else 0
    
    # Get the start and end times for the current window from your custom mechanism (np_pred.time[i])
    start_time, end_time = np_pred.time[i]
    
    # Find the indices in dfA that correspond to the current window
    # Make sure dfA['datetime'] is in the correct datetime format and np_pred.time[i] returns datetime objects or strings that can be converted to datetime
    dfA.loc[(dfA['temp_datetime'] >= pd.to_datetime(start_time)) & (dfA['temp_datetime'] <= pd.to_datetime(end_time)), 'flagged'] = flagged_value

# Identify and plot rectangles for continuous flagged intervals
dfA['flagged_diff'] = dfA['flagged'].diff()
start_flags = dfA[(dfA['flagged'] == 1) & (dfA['flagged_diff'] != 0)].index
end_flags = dfA[(dfA['flagged'] == 1) & (dfA['flagged_diff'].shift(-1) != 0)].index

for i, pred in enumerate(predictions[:10]):  # Adjust the range as necessary
    print(f"Max confidence for prediction {i}: {max(pred)}")
#################################################################################################################################################
# Functions for time series plot and labeling ui

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

labelsStartIndex, labelsEndIndex = calculate_label_indices(dfA)

def add_trace_to_layout(fig_layout, df, col_name, row, col):
    fig_layout.add_trace(go.Scatter(
        x=df['datetime'], y=df[col_name],
        mode='lines',
        name=col_name), row=row, col=col)

def update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, df):
    for i in range(len(labelsStartIndex)):
        start_idx = labelsStartIndex[i]
        end_idx = labelsEndIndex[i]
        label = df['label'].iloc[start_idx]

        fig_layout.add_trace(go.Scatter(
            x = df.loc[start_idx:end_idx,'datetime'],
            y = [1] * (end_idx - start_idx + 1),
            mode="lines",
            name=label,
            text=label,
            line_color=colorDict[label],
            textposition="top center",
            line_width=5,
            showlegend=False
        ), row=1, col=1)

def update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, df):
    for i in range(len(labelsStartIndex)):
        start_idx = labelsStartIndex[i]
        end_idx = labelsEndIndex[i]
        confidence = df['confidence'].iloc[start_idx]

        fig_layout.add_trace(go.Scatter(
            x = df.loc[start_idx:end_idx,'datetime'],
            y = [1] * (end_idx - start_idx + 1),  # Assuming labelLine is a constant value of 1
            mode="lines",
            name=confidence,
            text=confidence,
            line_color=confDict[confidence],
            textposition="top center",
            line_width=5,
            showlegend=False
        ), row=2, col=1)


def plot_raw_ts_graph():
    global fig_layout
    # Create a new figure layout for each call of plotGraph
    fig_layout = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[5,5,50],
        subplot_titles=("Label Line", "Confidence Line", "Raw Time-Series Data")
    )

    # Add raw time-series data to the plot
    for col in cols:
        add_trace_to_layout(fig_layout, dfA, col, 3, 1)

    # Update label and confidence lines
    labelsStartIndex, labelsEndIndex = calculate_label_indices(dfA)
    update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, dfA)
    update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, dfA)

    for start, end in zip(start_flags, end_flags):
        # Using datetime index to plot, ensure dfA['datetime'] is a datetime type
        fig_layout.add_vrect(
            x0=dfA.loc[start, 'datetime'], 
            x1=dfA.loc[end, 'datetime'],
            fillcolor="yellow",
            opacity=0.5,
            layer="below",
            line_width=0
        )

    # Update layout and return the figure
    fig_layout.update_layout(
        xaxis=dict(),  # end xaxis definition
        xaxis1_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=True,
        height=500,  # width=1000,
        legend_title_text="Sensors"
    )
    fig_layout.update_layout(height=500, #width=1000,
                  legend_title_text="Sensors")
    
    return fig_layout

########################################################################################################
## Functions for UMAP interface

def create_umap_figure():
    return px.scatter(
        embedding_df, 
        x='x', 
        y='y', 
        color=npInput_labels,
        hover_name=npInput_labels,
        color_discrete_map=color_discrete_map,
        custom_data=['index']
    ).update_layout(autosize=False, xaxis_title=None, yaxis_title=None, legend_title_text="Class", xaxis=dict(scaleanchor='y', scaleratio=1), yaxis=dict(scaleanchor='x',scaleratio=1,))

def rerender_umap(n_neighbors_value, min_dist_value):
    global embedding_df, umap_to_ts_mapping, predicted_labels
    reducer = umap.UMAP(n_neighbors=n_neighbors_value, n_components=2, min_dist=min_dist_value)
    embedding = reducer.fit_transform(predictions)

    embedding_df = pd.DataFrame(embedding, columns=['x', 'y'])
    embedding_df['index'] = np.arange(len(embedding))

    umap_to_ts_mapping = {umap_index: (window_start, window_end) for umap_index, (window_start, window_end) in enumerate(windows)}

    # Step 2: Assuming `predictions` are the output of your CNN model
    predicted_class_indices = np.argmax(predictions, axis=1)

    # Step 3: Convert numeric codes back to labels
    predicted_labels = np.array([num_to_label_dict[code] for code in predicted_class_indices])

    # Step 4: Create a mapping from windows to original data indices
    for umap_index, label in zip(embedding_df['index'], predicted_labels):
        window_start, window_end = umap_to_ts_mapping[umap_index]
        
        # Find rows where the 'datetime' column is within the window range
        mask = (dfA['temp_datetime'] >= window_start) & (dfA['temp_datetime'] <= window_end)
        
        # Assign the label to these rows
        dfA.loc[mask, 'PredictedLabel'] = label

    embedding_df['window_start'] = embedding_df['index'].apply(lambda idx: umap_to_ts_mapping[idx][0] if idx in umap_to_ts_mapping else None) # get times for each umap point

def create_time_series_figure(valid_features, cols, df):
    # Create a new figure
    fig = go.Figure()

    # Add each valid feature as a trace
    for feature in valid_features:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], 
                y=df[feature], 
                name=feature,
                visible="legendonly" if feature not in cols else True
            )
        )

    # Update layout
    fig.update_layout(
        xaxis_title="Datetime",
        legend=dict(
            title=dict(text="Features")
        ),
        clickmode='event+select'
    )

    return fig

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

def highlight_umap_point(umap_fig, selected_index, nearest_overall_index, nearest_same_type_index):
    # Define marker styles based on the source
    selected_marker = {
        'color': 'black', 'size': 10, 'symbol': 'circle-open',
        'line': {'color': 'black', 'width': 2}
    }
    nearest_overall_marker = {
        'color': 'red', 'size': 10, 'symbol': 'x',
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

def highlight_time_series_window(plot_fig, selected_id, df, new_np, embedding, npInput):
    selected_style = {'fillcolor': "grey", 'opacity': 0.5, 'line': {'color': 'white', 'width': 2}}
    nearest_overall_style = {'fillcolor': "red", 'opacity': 0.5, 'line': {'color': 'white', 'width': 2}}
    nearest_same_type_style = {'fillcolor': "purple", 'opacity': 0.5, 'line': {'color': 'white', 'width': 2}}

    # Highlight the selected point's time window & add dummy trace for legend
    start, end = new_np.time[selected_id]
    plot_fig.add_vrect(x0=start, x1=end, layer="below", **selected_style)
    plot_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(color=selected_style['fillcolor'],
                                            line=dict(color='black', width=0)),
                                name=f'Selected Point'))
    
    # Find and highlight nearest neighbors
    near_o, near_c = nearestNeighbor(embedding, selected_id, npInput)

    # Highlight the nearest neighbor overall's time window & add dummy trace for legend
    if near_o is not None:
        start, end = new_np.time[near_o]
        plot_fig.add_vrect(x0=start, x1=end, layer="below", **nearest_overall_style)
        plot_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                    marker=dict(color=nearest_overall_style['fillcolor'],
                                                line=dict(color='black', width=0)),
                                    name=f'Nearest Neighbor: Overall'))

    # Highlight the nearest neighbor of the same type's time window & add dummy trace for legend
    if near_c is not None and near_c != near_o:
        start, end = new_np.time[near_c]
        plot_fig.add_vrect(x0=start, x1=end, layer="below", **nearest_same_type_style)
        plot_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                    marker=dict(color=nearest_same_type_style['fillcolor'],
                                                line=dict(color='black', width=0)),
                                    name=f'Nearest Neighbor: Same Type'))

def find_umap_point_from_ts(clicked_datetime, windows):
    clicked_datetime = pd.to_datetime(clicked_datetime)
    # Iterate over each window to find where the clicked_datetime falls
    for umap_index, (start, end) in enumerate(windows):
        if start <= clicked_datetime <= end:
            return umap_index
    return None

def update_label(data_index, new_label, npInput, npInput_labels, label_num_dict):
    if data_index is not None and new_label in label_num_dict:
        # Update only the label for the specific data point
        new_label_code = label_num_dict[new_label]
        npInput[data_index] = new_label_code
        npInput_labels[data_index] = new_label

def update_subgraphs(graph1, graph2, graph3, id, df, new_np, embedding, npInput):
    # Extract indices for the nearest neighbors once to avoid recomputation
    nearest_overall_index, nearest_same_type_index = nearestNeighbor(embedding, id, npInput)

    # Prepare titles for clarity in the graphs
    titles = ["Selected Point", "Nearest Neighbor: Overall", "Nearest Neighbor: Same Type"]
    
    # Check if the nearest overall and same type are the same
    if nearest_overall_index == nearest_same_type_index:
        titles[2] = "Same as Nearest Overall"  # Change title for the third graph
        graph3.data = []  # Optionally clear any existing data
        graph3.update_layout(
            title=titles[2],
            showlegend=True,
            xaxis_title="Datetime",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            annotations=[{
                'text': "No distinct same-type neighbor; identical to nearest overall.",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 16}
            }]
        )

    # Prepare a list of indices corresponding to each graph
    indices = [id, nearest_overall_index, nearest_same_type_index]

    # Iterate over each graph object along with its corresponding data index and title
    for g, idx, title in zip([graph1, graph2, graph3], indices, titles):
        if idx is not None and title != "Same as Nearest Overall":
            # Extract the subset of dataframe for the given index
            g_data = df.loc[df["datetime"].between(new_np.time[idx][0], new_np.time[idx][1])]
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
            g.update_layout(
                showlegend=False
            )
            continue  # Already handled
            
        else:  # Handle no data case for other graphs
            g.data = []
            g.update_layout(
                title=f"No Data for {title}",
            )

        
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
        cleaned_fig = plot_raw_ts_graph()
    
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


########################################################################################################
# Set initial raw ts figure
initial_figure = plot_raw_ts_graph()
# Set initial umap
plotly_umap = create_umap_figure()

# set initial subgraphs
graph1 = px.scatter()
graph2 = px.scatter()
graph3 = px.scatter()

# Initialize app
app = dash.Dash(__name__)
app.title = 'Review Flagged Sections'

# App layout
app.layout = html.Div([
    # Container for TOP LEFT and TOP RIGHT columns
    html.Div([
        # TOP LEFT Column
        html.Div([
            dash_player.DashPlayer(id='video-player', url=video_path, controls=True, width='100%', height='400px'),
            html.Div(id="div-current-time"), 
            html.Div(html.Button('Reset Inputs', id='reset-button-2')),
            html.H5("Set Data/Video Offset:"), 
            dcc.Input(id='video-offset-input', type='text', placeholder="Input in Seconds", value="0"), 
            html.Br(),
            html.Div(id='video-offset-output'), 
            html.H5(children='''Sync Video to Data:'''), 
            html.Div(children='''Plot line on data graph at current time in video.'''),
            html.Button("Sync", id="button-sync-vid", n_clicks=0), 
            html.Br(), html.Br(),
        ], style={'padding': '20px', 'flex': 2}),

        # TOP RIGHT Column, adjusted to include two sections
        html.Div([
            # First part of TOP RIGHT
            html.Div([
                dcc.Graph(id='umap-graph', figure=plotly_umap),
                html.Div([
                    dcc.Markdown("""
                        **Adjust Parameters for UMAP:**
                    """),
                ]),
                html.Div([
                    html.Div([
                        html.Label('Number of Neighbors:', style={'font-weight': 'bold'}),
                        dcc.Input(
                            id='n_neighbors_input',
                            type='number',
                            value=15,  # Default starting value
                            min=2,  # UMAP's n_neighbors must be greater than 1
                        ),
                    ], style={'margin-bottom': '20px'}),

                    html.Div([
                        html.Label('Minimum Distance:', style={'font-weight': 'bold'}),
                        dcc.Slider(
                            id='min_dist_slider',
                            min=0.01,  # Minimum value of the slider
                            max=0.99, # Maximum value of the slider
                            step=0.01, # Increment step
                            value=0.1, # Default starting value
                            marks={
                                0.01: '0.01',
                                **{i / 100: '{:.1f}'.format(i / 100) for i in range(10, 90, 10)},
                                0.9: '0.9',
                                0.99: '0.99'
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], style={'margin-bottom': '20px', 'width': '50%'}),
                    
                    html.Button('Update UMAP', id='submit_button', n_clicks=0),
                ]),
                html.Br(),
                html.Div([dcc.Markdown("""**Click Data** Click on markers in the graph.""")]),
                html.Div([
                    html.Div([dcc.Dropdown(labelList, '', id='dropdown')], className="three columns"),
                    html.Div([html.Button('Add Label', id='button', n_clicks=0)], className="three columns")
                ], className="row"),
            ], style={'flex': 3}),  # Adjusted to take more space within the TOP RIGHT column

            # Newly positioned Div to the right of TOP RIGHT content
            html.Div([
                html.Div([dcc.Graph(id='graph1', figure=px.line(), style={'height': '300px','width': '300px'})], className="twelve columns"),
                html.Div([dcc.Graph(id='graph2', figure=px.line(), style={'height': '300px','width': '300px'})], className="twelve columns"),
                html.Div([dcc.Graph(id='graph3', figure=px.line(), style={'height': '300px','width': '300px'})], className="twelve columns"),
                dcc.Store(id='store_data', data=None, storage_type='memory')
            ], style={'flex': 1, 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-between'})
        ], style={'display': 'flex', 'padding': '20px', 'flex': 3}),  # Adjusted flex basis for the entire TOP RIGHT including the new section
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%'}),  # This ensures a flexible layout for the top part of the app

    # raw ts
    html.Div([
        # plot-clicked plot adjusted to take up more space
        html.Div([
            dcc.Graph(id='plot-clicked', figure=initial_figure)
        ], style={'flex': 3, 'display': 'inline-block', 'width': '75%'}),

        # User Input Fields container adjusted to take less space
        html.Div([
            html.H4("User Input Fields for Manually Adding a Label:"),
            "Start Time:", dcc.Input(id='start-input', type='text', placeholder="YYYY-MM-DD HH:MM:SS"),
            dcc.Checklist(id='start-checkbox', options=[{'label': 'Fill Start UI with Click', 'value': 'Checked'}], value=['Unchecked']), 
            html.Br(),
            "End Time:", dcc.Input(id='end-input', type='text', placeholder="YYYY-MM-DD HH:MM:SS"),
            dcc.Checklist(id='end-checkbox', options=[{'label': 'Fill End UI with Click', 'value': 'Checked'}], value=['Unchecked']), 
            html.Br(),
            "Labels:", dcc.Dropdown(labelList, placeholder='Select a Label', id='label-selection'), 
            html.Br(),
            "Degree of Confidence:", dcc.Dropdown(["High", "Medium", "Low", "Undefined"], placeholder='Select a Confidence Level', id='confidence-selection'), 
            html.Br(),
            html.Button('Update Graph', id='btn-manual-label', n_clicks=0),
        ], style={'padding': '20px', 'flex': 1, 'display': 'inline-block', 'width': '25%'}),
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%', 'align-items': 'flex-start'}),
])

########################################
# Callbacks for interactive components
########################################

# callback for printing current time under video
@app.callback(
    Output("div-current-time", "children"),
    Input("video-player", "currentTime")
)
def update_time(currentTime):
    update_time.data = currentTime
    return "Current Timestamp of Video: {}".format(currentTime)

@app.callback(
    [
        Output('umap-graph', 'figure'),
        Output('plot-clicked', 'figure'),
        Output('graph1', 'figure'),
        Output('graph2', 'figure'),
        Output('graph3', 'figure'),
        Output('store_data', 'data'),
        Output('video-player', 'seekTo')
     ],
    [
        Input('umap-graph', 'clickData'),
        Input('plot-clicked', 'clickData'),
        Input('graph1', 'clickData'),
        Input('graph2', 'clickData'),
        Input('graph3', 'clickData'),
        Input('button', 'n_clicks'),
        Input('submit_button', 'n_clicks'), # umap params
        Input('btn-manual-label', 'n_clicks'),
        Input('button-sync-vid', 'n_clicks')
     ],
    [
        State("dropdown", "value"),
        State("store_data", "data"),
        State('n_neighbors_input', 'value'), # umap params
        State('min_dist_slider', 'value'), # umap params
        State('start-input', 'value'),
        State('end-input', 'value'),
        State('label-selection', 'value'),
        State('confidence-selection', 'value'),
        State('video-offset-input', 'value'),
        State("video-player", "currentTime")
     ]
)
def update_combined(umap_clickData, plot_clickData, graph1_clickData, graph2_clickData, graph3_clickData, umap_param_submit, umap_label_n_clicks, btn_manual_label, btn_sync_vid, value, data,
                    n_neighbors, min_dist, start_time, end_time, label, confidence, video_offset, current_time):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]
    # intialize video seek time
    video_seek_time = dash.no_update

    umap_fig = create_umap_figure()
    umap_fig.update_layout(autosize=False, xaxis_title=None, yaxis_title=None, legend_title_text="Class")
    plot_fig = plot_raw_ts_graph()
    graph1 = px.scatter()
    graph2 = px.scatter()
    graph3 = px.scatter()

    if triggered == 'umap-graph':
        umap_fig = clear_highlights(umap_fig, "umap")
        plot_fig = clear_highlights(plot_fig, "ts")

        # Extract the index of the clicked point in the UMAP plot
        selected_index = umap_clickData["points"][0]["customdata"][0]
        nearest_overall_index, nearest_same_type_index = nearestNeighbor(embedding, selected_index, npInput)

        # Use the mapping to find the corresponding time series window
        window_start, window_end = umap_to_ts_mapping.get(selected_index, (None, None))
        if window_start and window_end:
            umap_fig = highlight_umap_point(umap_fig, selected_index, nearest_overall_index, nearest_same_type_index)
            highlight_time_series_window(plot_fig, selected_index, dfA, np_pred, embedding, npInput)
            update_subgraphs(graph1, graph2, graph3, selected_index, dfA, np_pred, embedding, npInput)

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

    if triggered == 'submit_button':
        rerender_umap(n_neighbors, min_dist)
        return [create_umap_figure()] + [dash.no_update]*6

    # Determine the source of the click
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

        clicked_datetime = pd.to_datetime(clickData['points'][0]['x'])
        umap_point_index = find_umap_point_from_ts(clicked_datetime, windows)
        nearest_overall_index, nearest_same_type_index = nearestNeighbor(embedding, umap_point_index, npInput)

        if umap_point_index is not None:
            window_start, window_end = umap_to_ts_mapping.get(umap_point_index, (None, None))
            if window_start and window_end:
                umap_fig = highlight_umap_point(umap_fig, umap_point_index, nearest_overall_index, nearest_same_type_index)
                highlight_time_series_window(plot_fig, umap_point_index, dfA, np_pred, embedding, npInput)
                update_subgraphs(graph1, graph2, graph3, umap_point_index, dfA, np_pred, embedding, npInput)
            
            # Calculate video seek time based on the time window start
            if window_start:
                video_seek_time = (window_start - video_start_time).total_seconds()

    if "button" == ctx.triggered_id:
        selected_index = umap_clickData["points"][0]["customdata"][0] if umap_clickData else None
        update_label(selected_index, value, npInput, npInput_labels, label_num_dict)
        npInput_labels_updated = np.array([num_to_label_dict[code] for code in npInput])
        color_discrete_map_updated = {label: base_color_map.get(label, '#000000') for label in np.unique(npInput_labels_updated)}
        umap_fig = px.scatter(
            embedding_df, x='x', y='y', color=npInput_labels_updated, hover_name=npInput_labels_updated,
            color_discrete_map=color_discrete_map_updated, custom_data=['index']
        ).update_layout(autosize=False, xaxis_title=None, yaxis_title=None, legend_title_text="Class")


    if triggered == 'btn-manual-label':
        # Update label and confidence data
        start = start_time
        end = end_time
        label = label
        confidence = confidence

        # Update DataFrame with new label and confidence data
        update_dataframe(dfA, start, end, label, confidence)

        # Recalculate label indices after updating DataFrame
        labelsStartIndex, labelsEndIndex = calculate_label_indices(dfA)

        # Replot the graph with updated data
        plot_fig = plot_raw_ts_graph()

    elif triggered == 'button-sync-vid':
        # Sync video logic
        # Get current video time and offset
        offset = int(video_offset)
        timestamp = int(update_time.data)

        # Calculate the synchronization point in data
        vid_to_data_sync = calculate_sync_point(dfA, offset, timestamp)

        # Update the graph with a sync point
        plot_fig = plotGraph_with_sync_point(vid_to_data_sync)

    return umap_fig, plot_fig, graph1, graph2, graph3, data, video_seek_time
    

def calculate_sync_point(df, offset, timestamp):
    # Calculate total elapsed time in seconds from the start of the data
    total_elapsed_time = offset + timestamp
    
    # Convert the 'datetime' column to a datetime object if not already
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Calculate the elapsed time for each row from the start of the data
    start_time = df['datetime'].iloc[0]
    df['elapsed_time'] = (df['datetime'] - start_time).dt.total_seconds()
    
    # Find the row in the dataframe that is closest to the total elapsed time
    closest_row_index = (df['elapsed_time'] - total_elapsed_time).abs().idxmin()
    sync_datetime = df['datetime'].iloc[closest_row_index]
    
    # Return the datetime in a string format suitable for plotting
    return sync_datetime.strftime('%Y-%m-%d %H:%M:%S')

def plotGraph_with_sync_point(sync_time):
    # Convert the 'datetime' column to a datetime object if not already
    dfA['datetime'] = pd.to_datetime(dfA['datetime'])

    # Reset and recreate the figure layout
    fig_layout = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[5,5,50],
        subplot_titles=("Label Line", "Confidence Line", "Raw Time-Series Data")
    )

    # Add time-series data and label/confidence lines
    for col in cols:
        add_trace_to_layout(fig_layout, dfA, col, 3, 1)
    update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, dfA)
    update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, dfA)

    # Add sync line
    fig_layout.add_vline(x=sync_time, line_width=2, line_dash="dash", line_color="red")
    fig_layout.add_annotation(x=sync_time, y=0.5, text="Video Sync", showarrow=False, yshift=10)

    # Update layout settings
    fig_layout.update_layout(
        xaxis=dict(),
        xaxis1_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=True,
        height=500,
        legend_title_text="Sensors"
    )

    return fig_layout

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

    # Update label and confidence data in the DataFrame
    df.loc[startIndex:endIndex, 'label'] = label
    df.loc[startIndex:endIndex, 'confidence'] = confidence

# fill start UI with clicked data
@app.callback(
    Output('start-input', 'value'),
    [Input('plot-clicked', 'clickData'),
     State('start-checkbox', 'value')])
def fill_start_time_with_click(clickData, checkbox_value):
    if clickData and 'Checked' in checkbox_value:
        xval = clickData['points'][0]['x']
        dt_val = xval.split(".")[0]
        return dt_val 
    return dash.no_update

# fill start UI with clicked data
@app.callback(
    Output('end-input', 'value'),
    [Input('plot-clicked', 'clickData'),
     State('end-checkbox', 'value')])  
def fill_end_time_with_click(clickData, checkbox_value):
    if clickData and 'Checked' in checkbox_value:
        xval = clickData['points'][0]['x']
        dt_val = xval.split(".")[0]  
        return dt_val  
    return dash.no_update

# run app in jupyter mode externally
# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=9122, jupyter_mode="external")