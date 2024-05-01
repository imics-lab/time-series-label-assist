# Dash framework and core components
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_player
import plotly.express as px
from dash_bootstrap_templates import load_figure_template

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
from datetime import timedelta

import sys
import os

from tyler_code import split, model
import umap
import math

####
# df = pd.read_csv('assets/manual_label_df.csv')
predictions = np.load('assets/predictions.npy')
dfA = pd.read_csv('assets/auto_label_df.csv')
video_path = 'assets/manual_video.mp4' # local video in asset directory
labelListDF = pd.read_csv('assets/label_list.csv')
labelList = list(labelListDF)
cols = list(pd.read_csv('assets/feature_cols.csv'))
window_size = 96
step_size = 32
colorList = ('#4363d8', '#e6194b', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
             '#000000')
# This assigns a specific color from colorList to each label in provided label list ** IF LABEL IS UNDEFINED COLOR IS BLACK**
colorDict = {label: ('#000000' if label == 'Undefined' else color) for label, color in zip(labelList, colorList)}

# Confidence list & dictionary
confidenceValues = ("High", "Medium", "Low", "Undefined")
confidenceColors = ('#3cb44b', '#ffe119', '#FF0000', '#000000')

confDict = dict(zip(confidenceValues, confidenceColors))

size = len(dfA)
# Makes list and assigns highest value to each index
labelLine = []
for i in range(size):
    labelLine.append(1)

class_index_to_label = {
    0: 'Downstairs',
    1: 'Jogging',
    2: 'Sitting',
    3: 'Standing',
    4: 'Upstairs',
    5: 'Walking'
}

# Find the index of the highest probability (predicted class index) in each prediction array
predicted_class_indices = [np.argmax(pred) for pred in predictions]

# Replicate the class indices according to the window and step size
expanded_class_indices = []
for i in range(len(predicted_class_indices)):
    expanded_class_indices.extend([predicted_class_indices[i]] * window_size)
    if i + 1 < len(predicted_class_indices):
        overlap = window_size - step_size
        expanded_class_indices = expanded_class_indices[:-overlap]

# Handle remaining rows, if any
remaining_rows = len(dfA) - len(expanded_class_indices)
if remaining_rows > 0:
    expanded_class_indices.extend([predicted_class_indices[-1]] * remaining_rows)

# Convert these indices to label names using the provided mapping
label_predictions = [class_index_to_label.get(index, "Undefined") for index in expanded_class_indices]

# Assign the expanded label predictions to the DataFrame
dfA['label'] = label_predictions
####

#@title Functions for plotting label line and confidence line
# for dfA
# currently only have label line
# fig_ll_rawTS = make_subplots(rows=2, cols=1,
#                     shared_xaxes=True,
#                     vertical_spacing=0.1,
#                     row_heights=[10, 200],
#                     subplot_titles=("Label Line", "Raw Time Series"),
#                     )
# #fig_ll_rawTS.update_layout(height=200)
# fix dfA to work in app
if "pred_labels" in dfA.columns:
    dfA['label'] = dfA['pred_labels']
    dfA = dfA.drop('pred_labels', axis=1)

if dfA.index.name != "datetime":
    dfA = dfA.set_index('datetime')

#print(dfA)
window_size = 96
step_size = 96

np_pred = split.TimeSeriesNP(window_size, step_size)
dfA.to_csv("dfA.csv")
np_pred.setArrays(dfA, encode = True, one_hot_encode=False ,labels =labelList)
npInput = np_pred.y
#print(npInput)

text_labels = []
for i in range(len(np_pred.y)):
  text_labels.append(np_pred.mapping[np_pred.y[i]])
#print(text_labels)
#new dictionary for altering labels
label_num_dict = {np_pred.mapping[k] : k for k in np_pred.mapping}
#print(label_num_dict)
#clear_output()
dfA = dfA.reset_index()

reducer = umap.UMAP(n_neighbors = 15, n_components =2)
# CHANGE 1
#embedding = reducer.fit_transform(predictions)
embedding = reducer.fit_transform(predictions)

def plotSubplotLabelLineRawTS():
    fig_ll_rawTS = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[20, 200],
                        subplot_titles=("Label Line", "Raw Time Series"),
                        )

    # Count number of labels
    tempDf = dfA.ne(dfA.shift())
    labelCount = tempDf.loc[tempDf.label == True, 'label'].count()
    # pd.set_option('display.max_rows', None)  # or 1000
    # print(tempDf['label'])
    # print(labelCount)

    # Make two lists to store indices
    labelsStartIndex = []
    labelsEndIndex = []
    # Start index at -1 to match indices
    index = -1
    # Take size to put into endIndex array to show last index
    size = len(tempDf['label'])

    # Loop through column to find Start Indices
    for element in tempDf['label']:
        if element == True:
            index+=1
            labelsStartIndex.append(index)
        else:
            index+=1
    # print("Start Indices", labelsStartIndex)

    # Loop through Start Indices list and get the index before next change
    for element in labelsStartIndex:
        labelsEndIndex.append(element - 1)
    # Remove first so we dont get the garbage value
    labelsEndIndex.pop(0)
    # Append size of column because thats last known label index
    labelsEndIndex.append(size)
    # print("End Indices", labelsEndIndex)

    # line with labels
    i = 0
    size = len(labelsStartIndex)
    size = int(size)
    currLabel = dfA['label']
    for x in range(size):
        fig_ll_rawTS.add_trace(go.Scatter(
            x = dfA.loc[labelsStartIndex[i]:labelsEndIndex[i],'datetime'],
            y = labelLine,
            mode="lines",
            name=currLabel.at[labelsStartIndex[i]],
            text=dfA.loc[labelsStartIndex[i]:labelsEndIndex[i],'label'],
            line_color=colorDict[currLabel.at[labelsStartIndex[i]]],
            textposition="top center",
            line_width=5,
            showlegend=False
            ), row=1, col=1)
        i+=1

    # Make figure for raw time series
    i = 0
    for element in cols:
      fig_ll_rawTS.add_trace(go.Scatter(x=dfA['datetime'], y=dfA[element],
                      mode='lines', # 'lines' or 'markers'
                      name=element), row=2, col=1)
      i+=1

    return fig_ll_rawTS

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

##############################

#@title Review Predicted Labels (DASH APP)
conf_thresh = 0.9
#print(dfA)
# append confidence column to dfA **NEW**
dfA_temp = dfA.copy()
dfA_temp["confidence"] = False
figt = px.scatter(x=dfA_temp["datetime"], y=dfA_temp["confidence"])

for i in range(len(predictions)):
    if max(predictions[i]) < conf_thresh:
        #print(max(predictions[i]))
        start = np_pred.time[i][0]
        end = np_pred.time[i][1]
        dfA_temp.loc[(dfA["datetime"] >= start) & (dfA_temp["datetime"] <= end), "confidence"] = True
        figt.add_vrect(
            x0=start,
            x1=end,
            fillcolor="yellow",
            opacity = 0.5,
            layer = "below",
        )
figt.update_traces(y=dfA_temp["confidence"])
#figt.show()

testModel = model.CNN()
testModel.setModel('assets\TWristARmodel.h5') # hardcode
testModel.only_test_data(np_pred.x, np_pred.y)

umap_pred = px.scatter(embedding, x=embedding[:,0], y=embedding[:,1], color = npInput,
                         range_color=[0, len(labelList)], hover_name=text_labels, color_continuous_scale=px.colors.sequential.Jet)

graph1 = px.scatter()
graph2 = px.scatter()
graph3 = px.scatter()
lineGraph = plotSubplotLabelLineRawTS()

for i in range(len(predictions)):
    if max(predictions[i]) < conf_thresh:
        #print(max(predictions[i]))
        start = np_pred.time[i][0]
        end = np_pred.time[i][1]
        lineGraph.add_vrect(
            x0 = start,
            x1 = end,
            fillcolor="yellow",
            opacity = 0.5,
            layer = "below",
            line_width = 0
        )

load_figure_template("bootstrap")
def layout():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    review_app = dash.Dash(__name__,external_stylesheets=external_stylesheets,)
    review_app.title = 'Review Flagged Labels'

    review_app.layout = html.Div([
        html.H3("Review Flagged Labels"),
        html.Br(),

        dcc.Markdown("**Video**"),
        # Video
        html.Div(className="row", children=[
            # video col 1
            html.Div(className="four columns", children=[
                dash_player.DashPlayer(id="video-player", url=video_path, controls=True,width="100%",height="100%"),
                html.Div(id='vid-sync-plot-dt-output'),
                html.Div(id='vid-sync-plot-offset-output'),
                html.Div(id='vid-sync-plot-timestamp-output'),
            ]),

            # UI column
            html.Div(className="four columns", children=[
                dcc.Markdown("""
                    **Set Data/Video Offset:**

                    Zero Offset: Both video and data start at zero.

                    Positive Offset: Data starts at zero, and video starts at offset.

                    Negative Offset: Data starts at offset, and video starts at zero.
                """),
                dcc.Input(id='video-offset-input', type='text', placeholder="Input in Seconds", value="0"),
                html.Div(id='vid-offset-output'),
                html.Br(),

                dcc.Markdown("""
                    **Sync Video to Data:**

                    Plot line on data graph at current time in video.
                """),
                html.Button("Sync", id="button-sync-vid", n_clicks=0),
                html.Br(), html.Br(),
            ]),

            # UI column 2
            html.Div(className="four columns", children=[
                dcc.Markdown("""
                    **Seek to Time in Video:**

                    E.g. If you want to go to 2 minute time stamp in video,
                      input in seconds, or "120
                """),
                dcc.Input(id='seek-input', type='text', placeholder="Input in Seconds"),
                html.Button("Seek", id="button-seek-to"),
                html.Div(id="div-current-time"),
                html.Br(),
                html.Div(html.Button('Reset Inputs', id='reset-button-2')),
                html.Br(),
            ])
        ]),
        html.Br(),

        # RAW TIME SERIES plot and 3 umap point plots
        html.Div([
            dcc.Markdown("""
            **Time-Series Plot**
            """),
            dcc.Graph(
                id='plot-clicked',
                figure = lineGraph
            ),
        ], ),

        html.Br(),
        html.Div([
            html.Div([
                dcc.Graph(id='graph1', figure=px.line())
            ], className="four columns"),

            html.Div([
                dcc.Graph(id='graph2', figure=px.line())
            ], className= "four columns"),

            html.Div([
                dcc.Graph(id='graph3', figure=px.line())
            ], className="four columns"),
        ], className="row"),
        dcc.Store(id='store_data', data = None, storage_type='memory'),

      # UI for manual label of RAW TS
      html.Div([
        dcc.Markdown("""
          **User Input Fields for Manually Adding a Label:**

          Format as YYYY-MM-DD HH:MM:SS E.g (2021-10-08 16:50:21)
        """),
      ]),
      html.Br(),
      html.Div(className="row", children=[
          # ui col 1
          html.Div(className="four columns", children=[
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
          ]),

          # ui col 2
          html.Div(className="four columns", children=[
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
                  "Flag: ",
                  dcc.Dropdown(["True", "False"],
                              placeholder = 'Select T (flag) or F (unflag)',
                              id='flag-selection'),
                  html.Div(id='flag-output')
              ]),
          ])
      ]),
      html.Button('Update Graph', id='btn-manual-label', n_clicks=0),
      html.Br(), html.Br(),

      # UMAP
      html.Div(className="row", children=[
          # UMAP col1,
          html.Div(className="eight columns", children=[
              dcc.Markdown("**UMAP**"),
              dcc.Graph(
                  id='umap-graph',
                  figure=umap_pred
              )
          ]),

          # UMAP UI, col 2
          html.Div(className="four columns", children=[
              html.Div([
                  dcc.Markdown("""
                      **Click Data**

                      Click on markets in the graph.
                  """),
              ]),
              html.Br(),
              html.Div([
                  html.Div([
                      dcc.Dropdown(labelList, '', id='dropdown')
                  ], className="three columns"),

                  html.Div([
                      html.Button('Add Label', id='umap-add-label-b', n_clicks=0)
                  ], className= "three columns"),
              ], className="row")
          ])
      ]),
    ])

    ############################################################################
    # callback for user input start time
    @review_app.callback(
        Output("start-output", "children"),
        Input('start-input', "value")
    )
    def update_start_time(start_value):
        update_start_time.data = start_value
        return "Start Value: {}".format(start_value)

    # callback for user input end time
    @review_app.callback(
        Output("end-output", "children"),
        Input('end-input', "value")
    )
    def update_end_time(end_value):
        update_end_time.data = end_value
        return "End Value: {}".format(end_value)

    # callback for user input label selection
    @review_app.callback(
        Output("label-output", "children"),
        Input('label-selection', "value")
    )
    def update_label(label_value):
        update_label.data = label_value
        return "Label Selection: {}".format(label_value)

    # callback for user input flag selection
    @review_app.callback(
        Output("flag-output", "children"),
        Input('flag-selection', "value")
    )
    def update_flag(flag_value):
        update_flag.data = flag_value
        return "Flag: {}".format(flag_value)

    # VIDEO
    # callback for user input video offset
    @review_app.callback(
        Output("vid-offset-output", "children"),
        Input('video-offset-input', "value")
    )
    def update_video_offset(videoOffset):
        update_video_offset.data = videoOffset
        return "Offset Value: {}".format(videoOffset)

    # callback for printing current time under video
    @review_app.callback(
        Output("div-current-time", "children"),
        Input("video-player", "currentTime")
    )
    def update_time(currentTime):
        update_time.data = currentTime
        return "Current Timestamp of Video: {}".format(currentTime)

    # umap & raw ts plot callbacks
    @review_app.callback(
        [Output('umap-graph', 'figure'),
         Output('plot-clicked', 'figure'),
         Output('graph1', 'figure'),
         Output('graph2', 'figure'),
         Output('graph3', 'figure'),
         Output('store_data', 'data')],
        Input('umap-graph', 'clickData'),
        Input('umap-add-label-b', 'n_clicks'),
        # added
        Input('btn-manual-label', 'n_clicks'),
        Input('button-sync-vid', 'n_clicks'),

        [State("dropdown", "value"),
         State("store_data","data")]
        )
    def update_app(clickData, n_clicks, clicks1, clicks2, value, data):

        #need to have some blank copies of graphs to avoid errors upon initial loading when nothing has been clicked
        umap_pred = px.scatter(embedding, x=embedding[:,0], y=embedding[:,1], color = npInput,
                                range_color=[0, len(labelList)], hover_name=text_labels, color_continuous_scale=px.colors.sequential.Jet)
        plot = plotSubplotLabelLineRawTS()
        for i in range(len(predictions)):
            if max(predictions[i]) < conf_thresh:
                #print(max(predictions[i]))
                start = np_pred.time[i][0]
                end = np_pred.time[i][1]
                plot.add_vrect(
                    x0 = start,
                    x1 = end,
                    fillcolor="yellow",
                    opacity = 0.5,
                    layer = "below",
                    line_width = 0
                )
        graph1 = px.line()
        graph2 = px.line()
        graph3 = px.line()
        id = None
        if "umap-graph" == ctx.triggered_id:
            data = clickData["points"][0]
            id = data['pointIndex']
            near_o, near_c = nearestNeighbor(embedding, id, npInput)
            umap_pred = px.scatter(embedding, x=embedding[:,0], y=embedding[:,1], color = npInput,
                         range_color=[0, len(labelList)], hover_name=text_labels, color_continuous_scale=px.colors.sequential.Jet)

            #add trace over selected point
            umap_pred.add_trace(
                go.Scatter(
                    mode='lines+markers+text',
                    x=[embedding[id][0]],
                    y=[embedding[id][1]],
                    marker=dict(
                        color='black',
                        size=10,
                        symbol="circle-open",
                        line=dict(
                            color='black',
                            width=2
                        ),
                    ),
                    name = "Selected Point",
                    showlegend=True
                )
            )

            #add point on UMAP for nearest overall neighbor
            umap_pred.add_trace(
                go.Scatter(
                    mode='lines+markers+text',
                    # text=["Nearest Neighbor Overall"],
                    # textposition = 'top center',
                    x=[embedding[near_o][0]],
                    y=[embedding[near_o][1]],
                    marker=dict(
                        color='red',
                        size=10,
                        symbol = "x",
                        line=dict(
                            color='black',
                            width=2
                        )
                    ),
                    name = "Nearest Neighbor: Overall",
                    showlegend=True
                )
            )



            #highlight segment of time-series for selected point
            plot = plotSubplotLabelLineRawTS()
            for i in range(len(predictions)):
                if max(predictions[i]) < conf_thresh:
                    #print(max(predictions[i]))
                    start = np_pred.time[i][0]
                    end = np_pred.time[i][1]
                    plot.add_vrect(
                        x0 = start,
                        x1 = end,
                        fillcolor="yellow",
                        opacity = 0.5,
                        layer = "below",
                        line_width = 0,
                    )
            plot.add_vrect(
              x0= np_pred.time[id][0], x1= np_pred.time[id][1],
              fillcolor="grey", opacity=0.5,
              layer="below", line_width=0,
            )

            plot.add_vrect(
              x0= np_pred.time[near_o][0], x1= np_pred.time[near_o][1],
              fillcolor="red", opacity=0.5,
              layer="below", line_width=0,
            )

            #modify subgraph
            g_1 = dfA.loc[dfA["datetime"].between(np_pred.time[id][0], np_pred.time[id][1])]
            graph1 = px.line(g_1, x = "datetime", y = cols)
            graph1.update_layout(title = "Selected Point", showlegend=False,xaxis_title="")

            g_2 = dfA.loc[dfA["datetime"].between(np_pred.time[near_o][0], np_pred.time[near_o][1])]
            graph2 = px.line(g_2, x = "datetime", y = cols)
            graph2.update_layout(title = "Nearest Neighbor: Overall",showlegend=False,xaxis_title="")

            graph3 = px.line()
            if near_o != near_c:
                umap_pred.add_trace(
                    go.Scatter(
                        mode='lines+markers+text',
                        # text=["Nearest Neighbor of Same Type"],
                        # textposition = 'top center',
                        x=[embedding[near_c][0]],
                        y=[embedding[near_c][1]],
                        marker=dict(
                            color='purple',
                            size=10,
                            symbol = "star",
                            line=dict(
                                color='black',
                                width=2
                            )
                        ),
                    name = "Nearest Neighbor: Same Type",
                    showlegend=True
                    )
                )

                plot.add_vrect(
                    x0= np_pred.time[near_c][0], x1= np_pred.time[near_c][1],
                    fillcolor="purple", opacity=0.5,
                    layer="below", line_width=0,
                )

                g_3 = dfA.loc[dfA["datetime"].between(np_pred.time[near_c][0], np_pred.time[near_c][1])]
                graph3 = px.line(g_3, x = "datetime", y = cols)
                graph3.update_layout(title = "Nearest Neighbor: Same Type", showlegend=False,xaxis_title="")
            else:
                graph3.update_yaxes(visible=False)
                graph3.update_xaxes(visible=False)

        if "umap-add-label-b" == ctx.triggered_id:
            # and data != None and value != ""
            npInput[data] =  label_num_dict[value]
            text_labels[data] = value
            # reset
            umap_pred = px.scatter(embedding, x=embedding[:,0], y=embedding[:,1], color = npInput,
                         range_color=[0, len(labelList)], hover_name=text_labels, color_continuous_scale=px.colors.sequential.Jet)

        # If update graph btn is clicke
        if "btn-manual-label" == ctx.triggered_id:
          start = update_start_time.data
          end = update_end_time.data
          label = update_label.data
          confidence = update_flag.data

          # Initialize a copy of dfA to manipulate
          label_df = dfA.copy()

          # Removes any decimals in datetime column, can assume label last at least a second
          # 2021-10-08 16:50:21.000000000 -> 16:50:21
          label_df['datetime'] = pd.to_datetime(label_df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
          # find start & end within 'datetime' column
          # Find start and end within 'datetime' column
          # Get Start indices and shove into list, First indice is where first label needs to go
          startIndices = label_df.index[label_df['datetime'] == start].tolist()
          startIndex = startIndices[0]
          # print("Starting Index", startIndex)
          # Get End indices and shove into list, Last indice is where labels need to end
          endIndices = label_df.index[label_df['datetime'] == end].tolist()
          endIndex = endIndices[-1]
          # print("Ending Index", endIndex)

          # Within df, look at 'label' column & assign label to range of startIndex to endIndex
          label_df.loc[startIndex:endIndex, 'label'] = label

          # label_df.label has our updated label column
          # Replaces original label column with the newly updated label column
          dfA.loc[:, 'label'] = label_df.label

          # replace confidence value inbetween start and end index
          label_df.loc[startIndex:endIndex, 'confidence'] = confidence
          dfA.loc[:, 'confidence'] = label_df.confidence

          # Update the graph
          plot = plotSubplotLabelLineRawTS()
          for i in range(len(predictions)):
              if max(predictions[i]) < conf_thresh:
                  #print(max(predictions[i]))
                  start = np_pred.time[i][0]
                  end = np_pred.time[i][1]
                  plot.add_vrect(
                      x0 = start,
                      x1 = end,
                      fillcolor="yellow",
                      opacity = 0.5,
                      layer = "below",
                      line_width = 0,
                  )
        # sync video button is clicked
        if "button-sync-vid" == ctx.triggered_id:
            # # create copy df to modify
            temp_df = dfA.copy()
            # Assume all datetime are in column named 'datetime'
            # Take first & last datetime from column
            firstDateTime = temp_df['datetime'][0]
            lastDateTime = temp_df['datetime'][len(temp_df['datetime']) - 1]
            firstDateTime = pd.to_datetime(firstDateTime)
            lastDateTime = pd.to_datetime(lastDateTime)

            # Convert first & last values to datetime, w/ unix specifications
            # cut down decimal
            firstDateTime = firstDateTime.strftime('%Y-%m-%d %H:%M:%S')
            # have to convert back to datetime
            firstDateTime = pd.to_datetime(firstDateTime)
            # unix specifications
            firstDateTime = pd.to_datetime(firstDateTime, unit='s', origin='unix')
            # cut down decimal
            lastDateTime = lastDateTime.strftime('%Y-%m-%d %H:%M:%S')
              # have to convert back to datetime
            lastDateTime = pd.to_datetime(lastDateTime)
            # unix specifications
            lastDateTime = pd.to_datetime(lastDateTime, unit='s', origin='unix')

            # Get unix time for first and last (datetime -> unix)
            firstUnixTime = (time.mktime(firstDateTime.timetuple()))
            lastUnixTime = (time.mktime(lastDateTime.timetuple()))

            # offset & timestamp retrieval
            # get from update_video_offset.data, strp by casting as int
            offset = int(update_video_offset.data)
            # get from update_time.data, , strp by casting as int
            timestamp = int(update_time.data)

            # if our offset is negative
            # be able to start data before its recorded
            if offset < 0:
              vid_to_data_sync = firstUnixTime + timestamp + abs(offset)

            else:
              # # get actual time after math in unix
              vid_to_data_sync = firstUnixTime + timestamp - offset
            # # # unix -> datetime, and strf down to whole second
            vid_to_data_sync = pd.to_datetime(vid_to_data_sync, unit='s', origin='unix')
            vid_to_data_sync = vid_to_data_sync.strftime('%Y-%m-%d %H:%M:%S')

            # reset graph
            plot = plotSubplotLabelLineRawTS()
            # plot at that x a red line
            plot.add_vline(
              x=vid_to_data_sync, line_width=2, line_dash="dash", line_color="red"
            )
            # label added at same x value
            plot.add_annotation(x=vid_to_data_sync, text="Video Sync", font=dict(size=15, color="black"),align="center")

        return umap_pred,plot,graph1,graph2,graph3,id

    # callback for seeking to timestamp in video from user inputted value
    @review_app.callback(
        Output("video-player", "seekTo"),
        [Input("button-seek-to", "n_clicks"),
        Input('seek-input', 'value')]
    )
    def set_seekTo(n_clicks, seek_value):
        if 'button-seek-to' == ctx.triggered_id:
          return seek_value

    # callback for resetting video UI
    @review_app.callback(
        [Output('video-offset-input', 'value'),Output('seek-input', 'value')],
        [Input('reset-button-2', 'n_clicks')])
    def reset_input(n_clicks):
        if n_clicks:
            return '0','0'
        else:
            return '0','0'

    # checkbox
    @review_app.callback(Output('start-cb-output', 'children'), [Input('start-checkbox', 'value')])
    def start_ui_fill(value):
        if len(value) == 2:
            start_ui_fill.data = value
            return "Start time fill toggled on"
        else:
            start_ui_fill.data = value
            return "Start time fill toggled off"

    # fill start UI with clicked data
    @review_app.callback(
        Output('start-input', 'value'),
        Input('plot-clicked', 'clickData'))
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
    @review_app.callback(Output('end-cb-output', 'children'), [Input('end-checkbox', 'value')])
    def end_ui_fill(value):
        if len(value) == 2:
            end_ui_fill.data = value
            return "End time fill toggled on"
        else:
            end_ui_fill.data = value
            return "End time fill toggled off"

    # fill start UI with clicked data
    @review_app.callback(
        Output('end-input', 'value'),
        Input('plot-clicked', 'clickData'))
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

    review_app.run_server(jupyter_mode='external')
layout()
