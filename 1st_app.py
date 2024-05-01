#@title Manual Labeling Interface

# Dash framework and core components
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output
import dash_player

# Dash Bootstrap Components for styling
import dash_bootstrap_components as dbc

# Plotly for interactive plotting
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Pandas for data manipulation
import pandas as pd

# Additional Python libraries
import time
from datetime import timedelta

import sys
import os
import json

##############################################################################################################################

# FETCH

df = pd.read_csv('assets/manual_label_df.csv')
labelListDF = pd.read_csv('assets/label_list.csv')
labelList = list(labelListDF)
video_path = 'assets/manual_video.mp4' # local video in asset directory

# Path to your JSON configuration
working_dir = os.getcwd()
assets_dir = os.path.join(working_dir, "assets")
config_path = os.path.join(assets_dir, 'config.json')

# Load the configuration
with open(config_path, 'r') as file:
    config = json.load(file)

# Extract lists from the configuration
valid_features = config["valid_features"]
features_to_omit = config["features_to_omit"]
cols = config["features_to_show"]

colorList = ('#4363d8', '#e6194b', '#3cb44b', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
             '#000000')

# This assigns a specific color from colorList to each label in provided label list ** IF LABEL IS UNDEFINED COLOR IS BLACK**
#colorDict = {label: ('#000000' if label == 'Undefined' else color) for label, color in zip(labelListDF, colorList)}

# Modify the dictionary comprehension to cycle through the color list (> 20 labels)
colorDict = {label: (colorList[i % len(colorList)] if label != 'Undefined' else '#000000') 
             for i, label in enumerate(labelListDF)}

# Confidence list & dictionary
confidenceValues = ("High", "Medium", "Low", "Undefined")
confidenceColors = ('#3cb44b', '#ffe119', '#FF0000', '#000000')

confDict = dict(zip(confidenceValues, confidenceColors))

##############################################################################################################################

# create UI fields for manually adding label & confidence level
manual_label_UI_fields = dbc.Card(
    [
      html.H4(children='''User Input Fields for Manually Adding a Label:'''),
      html.Div(children='''
        Format as YYYY-MM-DD HH:MM:SS E.g (2021-10-08 16:50:21)
      '''),
      html.Br(),
      
      html.Div([
          "Start Time: ",
          # Need to change ids, etc
          dcc.Input(id='start-input', 
                    type='text', 
                    placeholder = "YYYY-MM-DD HH:MM:SS"),
          dcc.Checklist(id='start-checkbox',
                    options=[{'label': 'Fill Start UI with Click', 'value': 'Checked'}],
                    value=['Unchecked']),
          html.Div(id='start-output'), 
          html.Div(id='start-cb-output'),
      ]),
      html.Br(),
      
      html.Div([
          "End   Time: ",
          # Need to change ids, etc
          dcc.Input(id='end-input',
                    type='text',
                    placeholder = "YYYY-MM-DD HH:MM:SS"),
          dcc.Checklist(id='end-checkbox',
                    options=[{'label': 'Fill End UI with Click', 'value': 'Checked'}],
                    value=['Unchecked']),
          html.Div(id='end-output'), 
          html.Div(id='end-cb-output'),
      ]),
      html.Br(),
      
      html.Div([
          "Labels: ",
          # Dummy values, need to get values from labelList csv
          dcc.Dropdown(labelList, 
                      placeholder = 'Select a Label',
                      id='label-selection'),
          # https://dash.plotly.com/dash-core-components/dropdown
          # this was for an output call back that prints curr value
          html.Div(id='label-output')
      ]),
      html.Br(),
      
      html.Div([
          "Degree of Confidence: ",
          dcc.Dropdown(["High", "Medium", "Low", "Undefined"], 
                      placeholder = 'Select a Confidence Level',
                      id='confidence-selection'),
          html.Div(id='confidence-output')
      ]),
      html.Br(),
      
      html.Button('Update Graph', id='btn-manual-label', n_clicks=0),
      html.Br(),
    ],
    body=True,
)

# create UI fields for video operations
video_UI_fields = dbc.Card(
    [
        # input field for offset
        html.H5(children='''Set Data/Video Offset:'''),
        html.Div(children='''Zero Offset: Both video and data start at zero.'''),
        html.Br(),
        html.Div(children='''Positive Offset: Data starts at zero, and video starts at offset."'''),
        html.Br(),
        html.Div(children='''Negative Offset: Data starts at offset, and video starts at zero."'''),
        html.Br(),
        dcc.Input(id='video-offset-input', type='text', placeholder = "Input in Seconds", value="0"),
        html.Div(id='vid-offset-output'),
        html.Br(),

        # sync video -> data
        html.H5(children='''Sync Video to Data:'''),
        html.Div(children='''Plot line on data graph at current time in video.'''),
        html.Button("Sync", id="button-sync-vid", n_clicks=0),
        html.Br(), html.Br(),


        # seek time input field
        html.H5(children='''Seek to Time in Video:'''),
        html.Div(children='''E.g. If you want to go to 2 minute time stamp in video, 
          input in seconds, or "120"'''),
        dcc.Input(id='seek-input', type='text', placeholder = "Input in Seconds"),
        html.Button("Seek", id="button-seek-to"),
        html.Div(id="div-current-time"),
        html.Div(html.Button('Reset Inputs', id='reset-button-2')),
        ],
    body=True,
)

####################################################################################################################

variable_name = ""
labelList = list(labelListDF)
size = len(df)

# set layout
fig_layout = make_subplots(rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[5,5,50],
                    subplot_titles=("Label Line", "Confidence Line", "Raw Time-Series Data"),
                    )
fig_layout.update_layout(height=10, width=1000)
# link x-axis for timeseries slider
fig_layout.update_xaxes(matches='x')
# theme stuff
#load_figure_template("bootstrap")
#fig_layout.update_layout(template="bootstrap")

# Makes list and assigns highest value to each index
labelLine = []
for i in range(size):
    labelLine.append(1)

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

labelsStartIndex, labelsEndIndex = calculate_label_indices(df)

def add_trace_to_layout(fig_layout, df, col_name, row, col):
    # Determine visibility based on whether col_name is in cols
    is_visible = True if col_name in cols else "legendonly"

    fig_layout.add_trace(go.Scatter(
        x=df['datetime'], y=df[col_name],
        mode='lines',
        name=col_name,
        visible=is_visible
    ), row=row, col=col)

def update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, df):
    for i in range(len(labelsStartIndex)):
        start_idx = labelsStartIndex[i]
        end_idx = labelsEndIndex[i]
        label = df['label'].iloc[start_idx]

        fig_layout.add_trace(go.Scatter(
            x = df.loc[start_idx:end_idx,'datetime'],
            y = [1] * (end_idx - start_idx + 1),  # Assuming labelLine is a constant value of 1
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


def plotGraph():
    global fig_layout
    # Create a new figure layout for each call of plotGraph
    fig_layout = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[5,5,50],
        subplot_titles=("Label Line", "Confidence Line", "Raw Time-Series Data")
    )

    # Add traces for all valid features
    for feature in valid_features:
        if feature not in features_to_omit:
            # Check if the feature is in cols to determine its initial visibility
            add_trace_to_layout(fig_layout, df, feature, 3, 1)

    # Update label and confidence lines
    labelsStartIndex, labelsEndIndex = calculate_label_indices(df)
    update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, df)
    update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, df)

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

# Set initial figure
initial_figure = plotGraph()

#fl_reset = go.Figure(fig_layout)

# Reference website https://dash.plotly.com/layout
# Build App
external_stylesheets = [dbc.themes.BOOTSTRAP]
#load_figure_template("bootstrap")

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Manually Label Time-Series Data'

app.layout = dbc.Container(
    [
        html.H2("Manually Label Raw Time-Series Data"),
        html.Hr(),
        # html.Div(children='''
        #   Dash: A web application framework for your data.
        # '''),
        html.Br(),
        
        # data
        dbc.Row(
            [
                dbc.Col(manual_label_UI_fields, md=4),
                # dbc.Col(dcc.Graph(id='graph-output',figure={}), md=8),
                dbc.Col(dbc.Card([
                    dcc.Graph(id='graph-output', figure=initial_figure)
                ]), 
                md=8),
            ],
            align="center", style={"height": "rem"}
        ),
    
        # video
        dbc.Row(
            [
                dbc.Col(video_UI_fields, md=4),
                dbc.Col(dbc.Card([
                    dash_player.DashPlayer(id = "video-player", url=video_path, controls=True)
                ])
                , md=4),
                html.Br(),
                # current vid vals
                html.Hr(),
                html.Div(id='vid-sync-plot-dt-output'),
                html.Div(id='vid-sync-plot-offset-output'),
                html.Div(id='vid-sync-plot-timestamp-output'),               
            ],
            align="center",
        ),
        html.Br(),
        html.Button('Save Changes', id='save_changes_button', n_clicks=0),
        html.Div(id='save_status'),
    ],
    fluid=True,
)

#####################################################
# callback for user input start time
@app.callback(
    Output("start-output", "children"),
    Input('start-input', "value")
)
def update_start_time(start_value):
    update_start_time.data = start_value
    return "Start Value: {}".format(start_value)

# callback for user input end time
@app.callback(
    Output("end-output", "children"),
    Input('end-input', "value")
)
def update_end_time(end_value):
    update_end_time.data = end_value
    return "End Value: {}".format(end_value)

# callback for user input label selection
@app.callback(
    Output("label-output", "children"),
    Input('label-selection', "value")
)
def update_label(label_value):
    update_label.data = label_value
    return "Label Selection: {}".format(label_value)

# callback for user input confidence selection
@app.callback(
    Output("confidence-output", "children"),
    Input('confidence-selection', "value")
)
def update_confidence_degree(confidence_value):
    update_confidence_degree.data = confidence_value
    return "Degree of Confidence: {}".format(confidence_value)

# callback for user input video offset
@app.callback(
    Output("vid-offset-output", "children"),
    Input('video-offset-input', "value")
)
def update_video_offset(videoOffset):
    update_video_offset.data = videoOffset
    return "Offset Value: {}".format(videoOffset)

# callback for printing current time under video
@app.callback(
    Output("div-current-time", "children"),
    Input("video-player", "currentTime")
)
def update_time(currentTime):
    update_time.data = currentTime
    return "Current Timestamp of Video: {}".format(currentTime)

@app.callback(
    Output('graph-output', 'figure'),
    [Input('btn-manual-label', 'n_clicks'),
    Input('button-sync-vid', 'n_clicks'),]
)
def updateGraph(btn1, btn2):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-manual-label':
        # Update label and confidence data
        start = update_start_time.data
        end = update_end_time.data
        label = update_label.data
        confidence = update_confidence_degree.data

        # Update DataFrame with new label and confidence data
        update_dataframe(df, start, end, label, confidence)

        # Recalculate label indices after updating DataFrame
        labelsStartIndex, labelsEndIndex = calculate_label_indices(df)

        # Replot the graph with updated data
        return plotGraph()

    elif button_id == 'button-sync-vid':
        # Sync video logic
        # Get current video time and offset
        offset = int(update_video_offset.data)
        timestamp = int(update_time.data)

        # Calculate the synchronization point in data
        vid_to_data_sync = calculate_sync_point(df, offset, timestamp)

        # Update the graph with a sync point
        return plotGraph_with_sync_point(vid_to_data_sync)

    else:
        return dash.no_update

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
        add_trace_to_layout(fig_layout, df, col, 3, 1)
    update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, df)
    update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, df)

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

# callback for seeking to timestamp in video from user inputted value
@app.callback(
    Output("video-player", "seekTo"), 
    [Input("button-seek-to", "n_clicks"),
    Input('seek-input', 'value')]
)
def set_seekTo(n_clicks, seek_value):
    if 'button-seek-to' == ctx.triggered_id:
        return seek_value

# callback for resetting video UI
@app.callback(
    [Output('video-offset-input', 'value'),Output('seek-input', 'value')],
    [Input('reset-button-2', 'n_clicks')])
def reset_input(n_clicks):
    if n_clicks:
        return '0','0'
    else:
        return '0','0'

# checkbox
@app.callback(Output('start-cb-output', 'children'), [Input('start-checkbox', 'value')])
def start_ui_fill(value):
    if len(value) == 2:
        start_ui_fill.data = value
        return "Start time fill toggled on"
    else:
        start_ui_fill.data = value
        return "Start time fill toggled off"

# fill start UI with clicked data
@app.callback(
    Output('start-input', 'value'),
    Input('graph-output', 'clickData'))
def st_x_value(clickData):
    try:
        if start_ui_fill.data.count('Checked') == 1:
            xval1 = clickData['points'][0]['x']
            st_x_value.data = xval1
            dt_val1 = xval1.split(".")[0]
            return str(dt_val1)
        else:
            return update_start_time.data
    except Exception as e:
        pass

# checkbox
@app.callback(Output('end-cb-output', 'children'), [Input('end-checkbox', 'value')])
def end_ui_fill(value):
    if len(value) == 2:
        end_ui_fill.data = value
        return "End time fill toggled on"
    else:
        end_ui_fill.data = value
        return "End time fill toggled off"

# fill start UI with clicked data
@app.callback(
    Output('end-input', 'value'),
    Input('graph-output', 'clickData'))
def ed_x_value(clickData):
    try:
        if end_ui_fill.data.count('Checked') == 1:
            xval2 = clickData['points'][0]['x']
            ed_x_value.data = xval2
            dt_val2 = xval2.split(".")[0]
            return str(dt_val2)
        else:
            return update_end_time.data
    except Exception as e:
        pass

@app.callback(
    Output('save_status', 'children'),  # You might want to add a component to indicate save status to the user
    [Input('save_changes_button', 'n_clicks')],
    prevent_initial_call=True
)
def save_changes(n_clicks):
    if n_clicks > 0:  # Check if the button has been clicked
        try:
            # Define the path where the updated DataFrame should be saved
            updated_df_path = os.path.join('assets', 'manual_label_df.csv')
            
            # Save the updated DataFrame to CSV
            df.to_csv(updated_df_path, index=False)  # Avoid saving with the index
            
            return 'Changes saved successfully!'  # Update the component with a success message
        except Exception as e:
            return f'Error saving changes: {e}'  # Update the component with an error message
    return ''  # Initial or default state with no message

# run app in jupyter mode externally
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8008, jupyter_mode="external")