# imports
import math
import os
import glob

import pandas as pd
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.models import load_model

import umap

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots

import dash
from dash import ctx
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import load_figure_template

from IPython.display import display
import ipywidgets as widgets

from tyler_code import split, model

working_dir = os.getcwd()

# necessary data loaded
# 1. model_select.value
# Path to the assets directory
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

# 2. new_np
# Loading the new_np object
with open(os.path.join('assets', 'new_np.pkl'), 'rb') as file:
    new_np = pickle.load(file)

# 3. df
# 4. labelList
# 5. cols
df = pd.read_csv('assets/manual_label_df.csv')
# Create a temporary column for the rounded datetimes
df['temp_datetime'] = pd.to_datetime(df['datetime']).dt.round('s')
labelListDF = pd.read_csv('assets/label_list.csv')
labelList = list(labelListDF)
cols = list(pd.read_csv('assets/feature_cols.csv'))
#6. window size
window_size = 96
step = 96

timestamps = pd.to_datetime(df['datetime'])  
windows = [(timestamps[i], timestamps[min(i + window_size - 1, len(timestamps) - 1)]) for i in range(0, len(timestamps), window_size)]
print("windows", windows)

testModel = model.CNN()
testModel.setModel(new_model)

#new dictionary for altering labels
label_num_dict = {new_np.mapping[k] : k for k in new_np.mapping}
print(label_num_dict)

colorList = ('#4363d8', '#e6194b', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
             '#000000')

# Create a color map: For each activity code, map it to a specific color. 
# Create a base color map for all labels except "Undefined"
base_color_map = {label: colorList[i] for i, label in enumerate(label_num_dict.keys()) if label != "Undefined"}

# Explicitly set the color for "Undefined" to black
base_color_map["Undefined"] = '#000000'

# Invert the label_num_dict to map from numeric codes to string labels
num_to_label_dict = {v: k for k, v in label_num_dict.items()}

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

load_figure_template("bootstrap")

npInput = new_np.y
# Map npInput numeric codes to string labels
npInput_labels = np.array([num_to_label_dict[code] for code in npInput])
color_discrete_map = {label: base_color_map.get(label, '#000000') for label in np.unique(npInput_labels)}

testModel.only_test_data(new_np.x, new_np.y)
predictions = testModel.model.predict(new_np.x, verbose=0, batch_size = 32)

reducer = umap.UMAP(n_neighbors = 15, n_components =2)
embedding = reducer.fit_transform(predictions)

embedding_df = pd.DataFrame(embedding, columns=['x', 'y'])
embedding_df['index'] = np.arange(len(embedding))

umap_to_ts_mapping = {umap_index: (window_start, window_end) for umap_index, (window_start, window_end) in enumerate(windows)}
print("umap/ts mapping", umap_to_ts_mapping)

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
    xaxis=dict(scaleanchor='y', scaleratio=1)
)

graph1 = px.scatter()
graph2 = px.scatter()
graph3 = px.scatter()
lineGraph = px.line(df, x = "datetime", y = cols)

def layout():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    umap_app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
    umap_app.title = 'UMAP'
    
    umap_app.layout = html.Div([
        html.H1("UMAP"),
        dcc.Graph(
            id='umap-graph',
            figure=plotly_umap
            ),
        html.Br(),
        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on markets in the graph.
            """),
        ]),        
        html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(labelList, '', id='dropdown')
            ], className="three columns"),

            html.Div([
                html.Button('Add Label', id='button', n_clicks=0)
            ], className= "three columns"),
        ], className="row")
            
        ]),
        html.Br(),
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
        dcc.Store(id='store_data', data = None, storage_type='memory')
    ])
    @umap_app.callback(
        [Output('umap-graph', 'figure'),
         Output('plot-clicked', 'figure'),
         Output('graph1', 'figure'),
         Output('graph2', 'figure'),
         Output('graph3', 'figure'),
         Output('store_data', 'data')],
        Input('umap-graph', 'clickData'),
        Input('plot-clicked', 'clickData'),
        Input('button', 'n_clicks'),
        [State("dropdown", "value"),
         State("store_data","data")]
        )
    def update_app(umap_clickData, plot_clickData, n_clicks, value, data):
        #need to have some blank copies of graphs to avoid errors upon initial loading when nothing has been clicked
        umap = px.scatter(
            embedding_df, 
            x='x', 
            y='y', 
            color=npInput_labels,
            hover_name=npInput_labels,
            color_discrete_map=color_discrete_map,
            custom_data=['index']
        )
        umap.update_layout(
            autosize=False,
            legend_title_text="Class",
            xaxis=dict(scaleanchor='y', scaleratio=1)
        )
        umap.update_layout(xaxis=dict(scaleanchor='y', scaleratio=1))

        plot = px.line(df, x = "datetime", y = cols)
        graph1 = px.line()
        graph2 = px.line()
        graph3 = px.line()
        id = None

        if "umap-graph" == ctx.triggered_id:
            data = umap_clickData["points"][0]
            id = data['customdata'][0]
            near_o, near_c = nearestNeighbor(embedding, id, npInput)

            umap = px.scatter(
                embedding_df, 
                x='x', 
                y='y', 
                color=npInput_labels,
                hover_name=npInput_labels,
                color_discrete_map=color_discrete_map,
                custom_data=['index']
            )
            umap.update_layout(
                autosize=False,
                legend_title_text="Class",
                xaxis=dict(scaleanchor='y', scaleratio=1)
            )  

            #add trace over selected point
            umap.add_trace(
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
            umap.add_trace(
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
            plot = px.line(df, x = "datetime", y = cols)
            plot.add_vrect(
              x0= new_np.time[id][0], x1= new_np.time[id][1],
              fillcolor="grey", opacity=0.5,
              layer="below", line_width=0,
            )
            plot.add_trace(
                  go.Scatter(
                      x=[None],  # Set to [None] so it doesn't appear on the plot
                      y=[None],  # Set to [None] so it doesn't appear on the plot
                      mode="markers",
                      marker=dict(size=10, color="grey"),
                      name="Selected Point",  # The label you want in the legend
                      visible="legendonly",  # Set visibility to legend only
                  )
              )
            
            plot.add_vrect(
              x0= new_np.time[near_o][0], x1= new_np.time[near_o][1],
              fillcolor="red", opacity=0.5,
              layer="below", line_width=0,
            )
            plot.add_trace(
                  go.Scatter(
                      x=[None],  # Set to [None] so it doesn't appear on the plot
                      y=[None],  # Set to [None] so it doesn't appear on the plot
                      mode="markers",
                      marker=dict(size=10, color="red"),
                      name="Nearest Neighbor: Overall",  # The label you want in the legend
                      visible="legendonly",  # Set visibility to legend only
                  )
              )
            
            #modify subgraph
            g_1 = df.loc[df["datetime"].between(new_np.time[id][0], new_np.time[id][1])]
            graph1 = px.line(g_1, x = "datetime", y = cols)
            graph1.update_layout(title = "Selected Point", showlegend=False,xaxis_title="")

            g_2 = df.loc[df["datetime"].between(new_np.time[near_o][0], new_np.time[near_o][1])]
            graph2 = px.line(g_2, x = "datetime", y = cols)
            graph2.update_layout(title = "Nearest Neighbor: Overall",showlegend=False,xaxis_title="")

            graph3 = px.line()
            if near_o != near_c:
                umap.add_trace(
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
                plot.add_trace(
                  go.Scatter(
                      x=[None],  # Set to [None] so it doesn't appear on the plot
                      y=[None],  # Set to [None] so it doesn't appear on the plot
                      mode="markers",
                      marker=dict(size=10, color="purple"),
                      name="Nearest Neighbor: Same Type",  # The label you want in the legend
                      visible="legendonly",  # Set visibility to legend only
                  )
              )
                plot.add_vrect(
                    x0= new_np.time[near_c][0], x1= new_np.time[near_c][1],
                    fillcolor="purple", opacity=0.5,
                    layer="below", line_width=0,
                )

                g_3 = df.loc[df["datetime"].between(new_np.time[near_c][0], new_np.time[near_c][1])]
                graph3 = px.line(g_3, x = "datetime", y = cols)
                graph3.update_layout(title = "Nearest Neighbor: Same Type", showlegend=False,xaxis_title="")
            else:
                graph3.update_yaxes(visible=False)
                graph3.update_xaxes(visible=False)

        if "plot-clicked" == ctx.triggered_id:
            clicked_datetime = plot_clickData['points'][0]['x']
            clicked_datetime = pd.to_datetime(clicked_datetime).round('s')  # Round to nearest second
            # Find the index of the clicked point in the df
            matching_indices = df.index[df['temp_datetime'] == clicked_datetime]

            if not matching_indices.empty:
                clicked_index = matching_indices[0]
                
                # Calculate which window the clicked index falls into
                window_start_index = (clicked_index // window_size) * window_size
                window_end_index = window_start_index + window_size - 1
                
                # Find the corresponding UMAP point index
                umap_point_index = window_start_index // window_size
                
                # Generate a new UMAP plot highlighting the corresponding point
                umap = go.Figure()
                umap = px.scatter(
                    embedding_df, 
                    x='x', 
                    y='y', 
                    color=npInput_labels,
                    hover_name=npInput_labels,
                    color_discrete_map=color_discrete_map,
                    custom_data=['index']
                )
                # Highlight the corresponding UMAP point
                if umap_point_index < len(embedding_df):
                    umap.add_trace(go.Scatter(
                        x=[embedding_df.iloc[umap_point_index]['x']],
                        y=[embedding_df.iloc[umap_point_index]['y']],
                        mode='markers+text',  # Add text mode if you wish to include annotations
                        marker=dict(
                            color='black',  # Use a contrasting color for the marker itself
                            size=15,  # Make the marker larger than the rest
                            symbol='star-diamond-open',  # Use a unique symbol
                            line=dict(
                                color='white',  # Use a contrasting outline color
                                width=2  # Adjust the width of the outline for visibility
                            )
                        ),
                        name='Time Series Sync',
                        text=['Time Series Sync'],  # This text can be adjusted or removed based on your preference
                        textposition='top center'  # Adjust text position as needed
                    ))
            umap.update_layout(title='UMAP Embedding')
            
            # print("Clicked datetime on raw time-series:", clicked_datetime)
            # print("Embedding index of the corresponding UMAP point:", umap_point_index)  
        
        if "button" == ctx.triggered_id:
            # and data != None and value != ""
            npInput[data] =  label_num_dict[value]
            npInput_labels[data] = value
            umap = px.scatter(
                embedding_df, 
                x='x', 
                y='y', 
                color=npInput_labels,
                hover_name=npInput_labels,
                color_discrete_map=color_discrete_map,
                custom_data=['index']
            )
            umap.update_layout(
                autosize=False,
                legend_title_text="Class",
                xaxis=dict(scaleanchor='y', scaleratio=1)
            )

        return umap,plot,graph1,graph2,graph3,id
    
    umap_app.run_server(debug=True, port=8052, jupyter_mode="external")
layout()