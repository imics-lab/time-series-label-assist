import dash
from dash import dcc, html, ctx, callback
from dash.dependencies import Input, Output, State
import dash_player
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import json
import flask

def layout():
    print("MANUAL LABELING TRIGGERED")
    if not flask.request:
        return html.Div()
    global labelList, valid_features, features_to_omit, cols, video_path, colorDict, confDict
    labelList = valid_features = features_to_omit = cols = video_path = None
    colorDict = confDict = {}

    # Set up paths and load configuration
    assets_dir = os.path.join(os.getcwd(), "assets")
    df = pd.read_csv(os.path.join(assets_dir, 'manual_label_df.csv'))
    labelListDF = pd.read_csv(os.path.join(assets_dir, 'label_list.csv'))
    labelList = list(labelListDF)

    config_path = os.path.join(assets_dir, 'config.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    valid_features = config["valid_features"]
    features_to_omit = config["features_to_omit"]
    cols = config["features_to_plot"]
    video_path = config["video_path"]
    
    # Assume undefined if confidence not in cols
    if "confidence" not in cols:
        df['confidence'] = "Undefined"

    colorList = ['#4363d8', '#e6194b', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                 '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                 '#000000']
    colorDict = {label: ('#000000' if label == 'Undefined' else color) for label, color in zip(labelList, colorList)}
    confidenceValues = ["High", "Medium", "Low", "Undefined"]
    confidenceColors = ['#3cb44b', '#ffe119', '#FF0000', '#000000']
    confDict = dict(zip(confidenceValues, confidenceColors))

    # Define callbacks here if needed
    # Build the layout dynamically
    layout = build_layout(df, labelList, valid_features, features_to_omit, cols, video_path, colorDict, confDict)
    return layout

# Retrieve start/end indices of all labels
def calculate_label_indices(df):
    tempDf = df.ne(df.shift())
    labelsStartIndex, labelsEndIndex = [], []
    index = -1
    for element in tempDf['label']:
        index += 1
        if element:
            labelsStartIndex.append(index)
            if labelsStartIndex and index > 0:
                labelsEndIndex.append(index - 1)
    labelsEndIndex.append(len(tempDf['label']) - 1)
    return labelsStartIndex, labelsEndIndex

def add_trace_to_layout(fig_layout, df, col_name, row, col):
    fig_layout.add_trace(go.Scatter(
        x=df['datetime'], y=df[col_name],
        mode='lines',
        name=col_name), row=row, col=col)

def update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, df, colorDict):
    for i in range(len(labelsStartIndex)):
        start_idx = labelsStartIndex[i]
        end_idx = labelsEndIndex[i]
        label = df['label'].iloc[start_idx]
        fig_layout.add_trace(go.Scatter(
            x=df.loc[start_idx:end_idx, 'datetime'],
            y=[1] * (end_idx - start_idx + 1),
            mode="lines",
            name=label,
            text=label,
            line_color=colorDict.get(label, '#000000'),
            textposition="top center",
            line_width=5,
            showlegend=False
        ), row=1, col=1)

def update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, df, confDict):
    for i in range(len(labelsStartIndex)):
        start_idx = labelsStartIndex[i]
        end_idx = labelsEndIndex[i]
        confidence = df['confidence'].iloc[start_idx]
        color = confDict.get(confidence, '#000000')
        fig_layout.add_trace(go.Scatter(
            x=df.loc[start_idx:end_idx, 'datetime'],
            y=[2] * (end_idx - start_idx + 1),
            mode="lines",
            name=f"Confidence: {confidence}",
            line=dict(color=color, width=4),
            showlegend=False
        ), row=2, col=1)

def plotGraph(df, cols, colorDict, confDict, labelsStartIndex=None, labelsEndIndex=None, selected_range=None, additional_shapes=[]):
    if labelsStartIndex is None or labelsEndIndex is None:
        labelsStartIndex, labelsEndIndex = calculate_label_indices(df)
    
    fig_layout = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[5, 5, 50], subplot_titles=("Label Line", "Confidence Line", "Raw Time-Series Data"))
    
    # Add traces
    for col in cols:
        add_trace_to_layout(fig_layout, df, col, 3, 1)
    update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, df, colorDict)
    update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, df, confDict)

    # If there's a selected range, add shapes
    if selected_range:
        start_date, end_date = selected_range
        fig_layout.add_shape(type='rect', xref='x', yref='paper', x0=start_date, x1=end_date, y0=0, y1=1, fillcolor='rgba(0, 0, 255, 0.2)', line={'width': 0})
    
    for shape in additional_shapes:
        fig_layout.add_shape(shape)
    
    fig_layout.update_layout(
        xaxis=dict(),
        xaxis1_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=True,
        height=500,
        legend_title_text="Sensors"
    )
    
    return fig_layout

def plotGraph_with_sync_point(df, cols, labelsStartIndex, labelsEndIndex, sync_time):
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
    update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, df, colorDict)
    update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, df, confDict)

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

    # Update the label only if provided
    if label is not None:
        df.loc[startIndex:endIndex, 'label'] = label

    # Update the confidence only if provided
    if confidence is not None:
        df.loc[startIndex:endIndex, 'confidence'] = confidence
    plotGraph(df, cols, colorDict=colorDict, confDict=confDict)

# Function to create the layout and also define callbacks dynamically
def build_layout(df, labelList, valid_features, features_to_omit, cols, video_path, colorDict, confDict):
    manual_label_UI_fields = dbc.Card([
        html.H4("User Input Fields for Manually Adding a Label:"),
        html.Div("Format as YYYY-MM-DD HH:MM:SS E.g (2021-10-08 16:50:21)"),
        html.Br(),
        html.Div([
            dcc.Checklist(
                id='fill-ui-checkbox',
                options=[{'label': 'Fill UI with selected range', 'value': 'fill-ui'}],
                value=[]
            ),
            html.Div(["Start Time: ", dcc.Input(id='start-input', type='text', placeholder="YYYY-MM-DD HH:MM:SS"), html.Div(id='start-output')]),
            html.Br(),
            html.Div(["End Time: ", dcc.Input(id='end-input', type='text', placeholder="YYYY-MM-DD HH:MM:SS"), html.Div(id='end-output')]),
            html.Br(),
            html.Div(["Labels: ", dcc.Dropdown(labelList, placeholder='Select a Label', id='label-selection'), html.Div(id='label-output')]),
            html.Br(),
            html.Div(["Degree of Confidence: ", dcc.Dropdown(["High", "Medium", "Low", "Undefined"], placeholder='Select a Confidence Level', id='confidence-selection'), html.Div(id='confidence-output')]),
            html.Br(),
            html.Button('Update Graph', id='btn-manual-label', n_clicks=0),
            html.Br(),
            html.Button('Save as CSV', id='btn-save-csv', n_clicks=0),
            html.Div(id='csv-save-output'),
        ])
    ])

    video_UI_fields = dbc.Card([
        html.H4("Set Data/Video Offset:"),
        html.Div([
            html.Strong("Zero Offset:"), html.Span(" Sync Start - Video and data begin together."),
            html.Br(),
            html.Strong("Positive Offset:"), html.Span(" Video Delay - Start video [offset] seconds after data."),
            html.Br(),
            html.Strong("Negative Offset:"), html.Span(" Data Delay - Start data [offset] seconds after video."),
            html.Br(),
            dcc.Input(id='video-offset-input', type='text', placeholder="Input in Seconds", value="0"),
            html.Div(id='vid-offset-output'),
            html.Br(),
            html.H4("Sync Video to Data:"), html.Div("Plot line on data graph at current time in video."),
            html.Button("Sync", id="button-sync-vid", n_clicks=0),
            html.Br(),
            html.H4("Seek to Time in Video:"), html.Div("E.g. If you want to go to 2 minute time stamp in video, input in seconds, or '120'"),
            dcc.Input(id='seek-input', type='text', placeholder="Input in Seconds"),
            html.Button("Seek", id="button-seek-to"),
            html.Div(id="div-current-time"),
            html.Div(html.Button('Reset Inputs', id='reset-button-2')),
        ])
    ])

    layout = dbc.Container([
        html.H2("Manually Label Raw Time-Series Data"),
        html.Hr(),
        dbc.Row([
            dbc.Col(manual_label_UI_fields, md=4),
            dbc.Col(dcc.Graph(id='graph-output', figure=plotGraph(df, cols, colorDict=colorDict, confDict=confDict)), md=8)  # Now directly using plotGraph
        ], align="center"),
        dbc.Row([
            dbc.Col(video_UI_fields, md=4),
            dbc.Col(dbc.Card([dash_player.DashPlayer(id="video-player", url=video_path, controls=True)]), md=8)
        ], align="center"),
        dcc.Store(id='data-store', data=df.to_json(date_format='iso', orient='split')),
    ], fluid=True)

    return layout

#####################################################
# callback for user input start time
@callback(
    Output("start-output", "children"),
    Input('start-input', "value")
)
def update_start_time(start_value):
    update_start_time.data = start_value
    return "Start Value: {}".format(start_value)

# callback for user input end time
@callback(
    Output("end-output", "children"),
    Input('end-input', "value")
)
def update_end_time(end_value):
    update_end_time.data = end_value
    return "End Value: {}".format(end_value)

# callback for user input label selection
@callback(
    Output("label-output", "children"),
    Input('label-selection', "value")
)
def update_label(label_value):
    update_label.data = label_value
    return "Label Selection: {}".format(label_value)

# callback for user input confidence selection
@callback(
    Output("confidence-output", "children"),
    Input('confidence-selection', "value")
)
def update_confidence_degree(confidence_value):
    update_confidence_degree.data = confidence_value
    return "Degree of Confidence: {}".format(confidence_value)

# save df
@callback(
    Output('csv-save-output', 'children'),
    Input('btn-save-csv', 'n_clicks'),
    State('data-store', 'data')
)
def save_csv(n_clicks, jsonified_df):
    if n_clicks > 0:
        df = pd.read_json(jsonified_df, orient='split')
        df.to_csv('assets/manual_label_df.csv', index=False)
        return 'Data saved as CSV'
    return ''

# callback for user input video offset
@callback(
    Output("vid-offset-output", "children"),
    Input('video-offset-input', "value")
)
def update_video_offset(videoOffset):
    update_video_offset.data = videoOffset
    return "Offset Value: {}".format(videoOffset)

# callback for printing current time under video
@callback(
    Output("div-current-time", "children"),
    Input("video-player", "currentTime")
)
def update_time(currentTime):
    update_time.data = currentTime
    return "Current Timestamp of Video: {}".format(currentTime)

def calculate_sync_point(df, offset, timestamp):
    # Convert the first datetime in df to a timezone-naive datetime object (if not already)
    firstDateTime = pd.to_datetime(df['datetime'].iloc[0]).to_pydatetime()

    # Calculate the difference in seconds from the first datetime
    df['timedelta'] = (pd.to_datetime(df['datetime']) - firstDateTime).dt.total_seconds()

    # Calculate the synchronization point in seconds
    sync_seconds = timestamp + offset

    # Find the closest datetime in df to the synchronization point
    # This assumes df['timedelta'] is sorted; if not, sort it first
    closest_row = df.iloc[(df['timedelta'] - sync_seconds).abs().argsort()[:1]]

    # Extract the datetime from the closest row
    sync_time = closest_row['datetime'].iloc[0]

    # Convert sync_time to a datetime object and format it
    formatted_sync_time = pd.to_datetime(sync_time).strftime('%Y-%m-%d %H:%M:%S')

    return formatted_sync_time

# callback for seeking to timestamp in video from user inputted value
@callback(
    Output("video-player", "seekTo"), 
    [Input("button-seek-to", "n_clicks"),
    Input('seek-input', 'value')]
)
def set_seekTo(n_clicks, seek_value):
    if 'button-seek-to' == ctx.triggered_id:
        return seek_value

# callback for resetting video UI
@callback(
    [Output('video-offset-input', 'value'),Output('seek-input', 'value')],
    [Input('reset-button-2', 'n_clicks')])
def reset_input(n_clicks):
    if n_clicks:
        return '0','0'
    else:
        return '0','0'

@callback(
    Output('start-input', 'value'),
    Input('graph-output', 'relayoutData'),
    State('fill-ui-checkbox', 'value')
)
def update_start_input(relayoutData, fill_ui_value):
    if relayoutData and 'xaxis.range[0]' in relayoutData:
        if 'fill-ui' in fill_ui_value:
            start_date = relayoutData['xaxis.range[0]']
            return start_date
    return ''

@callback(
    Output('end-input', 'value'),
    Input('graph-output', 'relayoutData'),
    State('fill-ui-checkbox', 'value')
)
def update_end_input(relayoutData, fill_ui_value):
    if relayoutData and 'xaxis.range[1]' in relayoutData:
        if 'fill-ui' in fill_ui_value:
            end_date = relayoutData['xaxis.range[1]']
            return end_date
    return ''

@callback(
    [Output('graph-output', 'figure'),
     Output('data-store', 'data')],  # Include this to update the stored DataFrame
    [Input('btn-manual-label', 'n_clicks'),
     Input('button-sync-vid', 'n_clicks'),
     Input('graph-output', 'relayoutData'),
     Input('data-store', 'data')],
    [State('fill-ui-checkbox', 'value'),
     State('start-input', 'value'),
     State('end-input', 'value'),
     State('label-selection', 'value'),
     State('confidence-selection', 'value')]
)
def combined_callback(btn_manual_label, btn_sync_vid, relayoutData, jsonified_df, fill_ui_value, start_input, end_input, label_selection, confidence_selection):
    df = pd.read_json(jsonified_df, orient='split')
    labelsStartIndex, labelsEndIndex = calculate_label_indices(df)

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'btn-manual-label':
        if start_input and end_input:
            start = pd.to_datetime(start_input)
            end = pd.to_datetime(end_input)
            df.loc[(df['datetime'] >= start) & (df['datetime'] <= end), 'label'] = label_selection
            df.loc[(df['datetime'] >= start) & (df['datetime'] <= end), 'confidence'] = confidence_selection
            updated_figure = plotGraph(df, cols, colorDict, confDict, labelsStartIndex, labelsEndIndex)
            return updated_figure, df.to_json(date_format='iso', orient='split')  # Return updated DataFrame in JSON format

    elif trigger_id == 'button-sync-vid':
        offset = int(update_video_offset.data)
        timestamp = int(update_time.data)
        vid_to_data_sync = calculate_sync_point(df, offset, timestamp)
        updated_figure = plotGraph_with_sync_point(df, cols, labelsStartIndex, labelsEndIndex, vid_to_data_sync)
        return updated_figure, df.to_json(date_format='iso', orient='split')

    elif trigger_id == 'graph-output':
        if 'fill-ui' in fill_ui_value and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
            start_date = relayoutData['xaxis.range[0]']
            end_date = relayoutData['xaxis.range[1]']
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
            updated_figure = plotGraph(df, cols, colorDict, confDict, labelsStartIndex, labelsEndIndex, selected_range=(start_date, end_date), additional_shapes=[highlight_shape])
            return updated_figure, df.to_json(date_format='iso', orient='split')

    # Default case to handle general update or initialization
    return plotGraph(df, cols, colorDict, confDict, labelsStartIndex, labelsEndIndex), jsonified_df  # Return the original data if no updates are done