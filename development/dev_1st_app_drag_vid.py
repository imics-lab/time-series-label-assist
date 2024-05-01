#@title Manual Labeling Interface

# Dash framework and core components
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
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


##############################################################################################################################

# FETCH
df = pd.read_csv('assets/manual_label_df.csv')
labelListDF = pd.read_csv('assets/label_list.csv')
labelList = list(labelListDF)
video_path = 'assets/manual_video.mp4' # local video in asset directory
cols = list(pd.read_csv('assets/feature_cols.csv'))

colorList = ('#4363d8', '#e6194b', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
             '#000000')
# This assigns a specific color from colorList to each label in provided label list ** IF LABEL IS UNDEFINED COLOR IS BLACK**
colorDict = {label: ('#000000' if label == 'Undefined' else color) for label, color in zip(labelListDF, colorList)}

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
        dcc.Checklist(
            id='fill-ui-checkbox',
            options=[
                {'label': 'Fill UI with selected range', 'value': 'fill-ui'},
            ],
            value=[],
        )
      ]),
      html.Div([
          "Start Time: ",
          # Need to change ids, etc
          dcc.Input(id='start-input', 
                    type='text', 
                    placeholder = "YYYY-MM-DD HH:MM:SS"),
          html.Div(id='start-output'), 
      ]),
      html.Br(),
      
      html.Div([
          "End   Time: ",
          # Need to change ids, etc
          dcc.Input(id='end-input',
                    type='text',
                    placeholder = "YYYY-MM-DD HH:MM:SS"),
          html.Div(id='end-output'), 
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
      html.Button('Save as CSV', id='btn-save-csv', n_clicks=0),
      html.Div(id='csv-save-output'),
    ],
    body=True,
)

# create UI fields for video operations
video_UI_fields = dbc.Card(
    [
    # input field for offset
    html.H4(children='Set Data/Video Offset:'),  # Larger heading for the main title
    html.Div([
        html.Div([
            html.Strong('Zero Offset:'),  # Strong tag for offset title
            html.Span(' Sync Start - Video and data begin together.')
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Strong('Positive Offset:'),
            html.Span(' Video Delay - Start video [offset] seconds after data.')
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Strong('Negative Offset:'),
            html.Span(' Data Delay - Start data [offset] seconds after video.')
        ], style={'margin-bottom': '10px'})
    ]),
        dcc.Input(id='video-offset-input', type='text', placeholder = "Input in Seconds", value="0"),
        html.Div(id='vid-offset-output'),
        html.Br(),

        # sync video -> data
        html.H4(children='''Sync Video to Data:'''),
        html.Div(children='''Plot line on data graph at current time in video.'''),
        html.Button("Sync", id="button-sync-vid", n_clicks=0),
        html.Br(), html.Br(),


        # seek time input field
        html.H4(children='''Seek to Time in Video:'''),
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
        confidence = df['confidence'].iloc[start_idx]  # Assuming uniform confidence within a segment

        # Determine the color based on the confidence level
        color = confDict.get(confidence, '#000000')  # Default to black if confidence level is not found

        fig_layout.add_trace(go.Scatter(
            x=df.loc[start_idx:end_idx, 'datetime'],
            y=[2] * (end_idx - start_idx + 1),  # Plot on a separate line for visibility
            mode="lines",
            name=f"Confidence: {confidence}",
            line=dict(color=color, width=4),  # Use the determined color
            showlegend=False
        ), row=2, col=1)

def plotGraph(df, selected_range=None, additional_shapes=[]):
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
        add_trace_to_layout(fig_layout, df, col, 3, 1)

    # Update label and confidence lines
    labelsStartIndex, labelsEndIndex = calculate_label_indices(df)
    update_label_lines(fig_layout, labelsStartIndex, labelsEndIndex, df)
    update_confidence_lines(fig_layout, labelsStartIndex, labelsEndIndex, df)

    # allow zoom
    fig_layout.update_layout(uirevision=True)

    # Turn off y-axis ticks for subplots 1 and 2
    fig_layout.update_yaxes(showticklabels=False, row=1, col=1)
    fig_layout.update_yaxes(showticklabels=False, row=2, col=1)
    
    # Add the highlight for the selected range if provided
    if selected_range:
        start_date, end_date = selected_range
        fig_layout.add_shape(
            type='rect',
            xref='x',
            yref='paper',
            x0=start_date,
            x1=end_date,
            y0=0,
            y1=1,
            fillcolor='rgba(0, 0, 255, 0.2)',
            line={'width': 0}
        )
        # snap to original
        fig_layout.update_layout(uirevision=False)

    # Add additional shapes (highlighted area) if provided
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

    # Update the label only if provided
    if label is not None:
        df.loc[startIndex:endIndex, 'label'] = label

    # Update the confidence only if provided
    if confidence is not None:
        df.loc[startIndex:endIndex, 'confidence'] = confidence
    plotGraph(df)
    
# Set initial figure
initial_figure = plotGraph(df)

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

# save df
@app.callback(
    Output('csv-save-output', 'children'),
    Input('btn-save-csv', 'n_clicks')
)
def save_csv(n_clicks):
    if n_clicks > 0:
        df.to_csv('assets/manual_label_df.csv', index=False)
        return 'Data saved as CSV'
    return ''

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

@app.callback(
    Output('start-input', 'value'),
    Input('graph-output', 'relayoutData'),
    State('fill-ui-checkbox', 'value')
)
def update_start_input(relayoutData, fill_ui_value):
    if relayoutData and 'xaxis.range[0]' in relayoutData:
        if 'fill-ui' in fill_ui_value:
            start_date = relayoutData['xaxis.range[0]']
            return start_date
    return None

@app.callback(
    Output('end-input', 'value'),
    Input('graph-output', 'relayoutData'),
    State('fill-ui-checkbox', 'value')
)
def update_end_input(relayoutData, fill_ui_value):
    if relayoutData and 'xaxis.range[1]' in relayoutData:
        if 'fill-ui' in fill_ui_value:
            end_date = relayoutData['xaxis.range[1]']
            return end_date
    return None

@app.callback(
    Output('graph-output', 'figure'),
    [Input('btn-manual-label', 'n_clicks'),
     Input('button-sync-vid', 'n_clicks'),
     Input('graph-output', 'relayoutData')],
    [State('fill-ui-checkbox', 'value'),
     State('start-input', 'value'),
     State('end-input', 'value'),
     State('label-selection', 'value'),
     State('confidence-selection', 'value')]
)
def combined_callback(btn_manual_label, btn_sync_vid, relayoutData, fill_ui_value, start_input, end_input, label_selection, confidence_selection):
    ctx = dash.callback_context

    # Determine which input was triggered
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'btn-manual-label':
        # Convert start and end input values to datetime if they're not None
        if start_input and end_input:
            start = pd.to_datetime(start_input)
            end = pd.to_datetime(end_input)
        else:
            start, end = None, None

        # Use the label and confidence directly from the callback's state
        label = label_selection
        confidence = confidence_selection

        # Update DataFrame with new label and confidence data
        # Assuming update_dataframe updates your DataFrame based on start/end times, label, and confidence
        update_dataframe(df, start, end, label, confidence)

        # Recalculate label indices after updating DataFrame
        labelsStartIndex, labelsEndIndex = calculate_label_indices(df)

        # Replot the graph with updated data
        return plotGraph(df)

    elif trigger_id == 'button-sync-vid':
        # Sync video logic
        # Get current video time and offset
        offset = int(update_video_offset.data)
        timestamp = int(update_time.data)

        # Calculate the synchronization point in data
        vid_to_data_sync = calculate_sync_point(df, offset, timestamp)

        # Update the graph with a sync point
        return plotGraph_with_sync_point(vid_to_data_sync)

    elif trigger_id == 'graph-output':
        # Check if the fill-ui-checkbox is checked and there is a selected range
        if 'fill-ui' in fill_ui_value and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
            start_date = relayoutData['xaxis.range[0]']
            end_date = relayoutData['xaxis.range[1]']

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

            return plotGraph(df, selected_range=(start_date, end_date), additional_shapes=[highlight_shape])
        
    return plotGraph(df)

# run app in jupyter mode externally
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8505, jupyter_mode="external")