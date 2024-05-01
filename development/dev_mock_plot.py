import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd  # Assuming pandas is used to handle the DataFrame

#################################################################
df = pd.read_csv('assets/manual_label_df.csv')
cols = list(pd.read_csv('assets/feature_cols.csv'))
labelListDF = pd.read_csv('assets/label_list.csv')
labelList = list(labelListDF)

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
                             
colorList = ('#4363d8', '#e6194b', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
             '#000000')
# This assigns a specific color from colorList to each label in provided label list ** IF LABEL IS UNDEFINED COLOR IS BLACK**
colorDict = {label: ('#000000' if label == 'Undefined' else color) for label, color in zip(labelListDF, colorList)}

df['datetime'] = pd.to_datetime(df['datetime'])

# Initialize 'confidence' to 'High'
df['confidence'] = 'High'

# Define your low confidence intervals
low_conf_intervals = [(df['datetime'].iloc[100], df['datetime'].iloc[200]), 
                      (df['datetime'].iloc[1800], df['datetime'].iloc[1900])]

# Set confidence to 'Low' within specified intervals
for start, end in low_conf_intervals:
    df.loc[(df['datetime'] >= start) & (df['datetime'] <= end), 'confidence'] = 'Low'

#################################################################

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='timeseries-with-labels'),
])

@app.callback(
    Output('timeseries-with-labels', 'figure'),
    Input('timeseries-with-labels', 'id')
)
def update_graph(dummy):
    fig = go.Figure()
    
    # Add the raw time series data
    for col in cols:
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df[col],
            mode='lines',
            name=col,
            customdata=df[['label', 'confidence']],
            hovertemplate="Label: %{customdata[0]}<br>Confidence: %{customdata[1]}<extra></extra>",
        ))

    # Add rectangles and hover traces for labels
    for start_idx, end_idx in zip(labelsStartIndex, labelsEndIndex):
        start_date, end_date = df['datetime'].iloc[start_idx], df['datetime'].iloc[end_idx]
        label = df['label'].iloc[start_idx]
        confidence = df['confidence'].iloc[start_idx]

        color = colorDict.get(label, '#000000')
        midpoint_index = start_idx + (end_idx - start_idx) // 2
        midpoint_date = df['datetime'].iloc[midpoint_index]
        annotation_y = max(df[cols].max()) * 1.25

        hover_text = f"{label}" if confidence is None else f"{label}: {confidence}"

        ###############################################################################################    
        fig.add_shape(type="rect",
                      x0=start_date, y0=min(df[cols].min()), x1=end_date, y1=max(df[cols].max()),
                      fillcolor=color, opacity=0.3, layer="below", line_width=0.5)

        # Add annotation for the label, excluding confidence if label is "Undefined"
        if label == "Undefined":
            annotation_text = f"{label}"
        else:
            annotation_text = f"{label}: {confidence}"

        fig.add_annotation(x=midpoint_date, y=annotation_y,
                           text=annotation_text,
                           showarrow=False, font=dict(color='#000000'))
        ###############################################################################################  

    # Define and add fake low confidence intervals
    low_conf_intervals = [(df['datetime'].iloc[100], df['datetime'].iloc[200]), 
                          (df['datetime'].iloc[1800], df['datetime'].iloc[1900])]
    for start_time, end_time in low_conf_intervals:
        fig.add_shape(type="rect",
                      x0=start_time, y0=0, x1=end_time, y1=1,
                      xref="x", yref="paper", fillcolor=None, opacity=1,                
                      layer="below", line_width=1.5, line_color="black")
        # Add corresponding annotation manually
        fig.add_annotation(
            x=start_time, 
            y=1,  # Annotation at middle height of the graph
            xref="x", 
            yref="paper",
            text="Low Confidence", 
            showarrow=False,
            yshift=20,  # Shift the text up a bit so it doesn't sit directly on the rectangle
            #bgcolor="white",
            opacity=1
        )
    fig.update_layout(xaxis_title="datetime", yaxis_title="values", title="Labels Over Sensor Data",
                      hovermode="x unified", showlegend=True)
    
    # Labels in Legend
    # for label, color in colorDict.items():
    #     if label != "Undefined":
    #         fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
    #                                 name=f"Label: {label}",
    #                                 line=dict(color=color), showlegend=True,))


    fig.write_html("dev_UI1_mock.html")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8500)
