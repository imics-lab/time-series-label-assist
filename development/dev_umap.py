import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

# Generate synthetic data
data, labels = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
df = pd.DataFrame(data, columns=['x', 'y'])
df['label'] = labels

# Decide on the percentage of mislabeled points
mislabeled_percentage = 5  # Mislabel 5% of the points

# Calculate number of points to mislabel
num_points = len(df)
num_mislabeled = int((mislabeled_percentage / 100) * num_points)

# Randomly select indices to mislabel
np.random.seed(3)  # Optional: for reproducible results
mislabeled_indices = np.random.choice(num_points, size=num_mislabeled, replace=False)

# Mislabel the selected points
for index in mislabeled_indices:
    current_label = df.loc[index, 'label']
    possible_labels = set(df['label'].unique()) - {current_label}  # Exclude the current label
    new_label = np.random.choice(list(possible_labels))
    df.loc[index, 'label'] = new_label

# Initialize Dash app
app = dash.Dash(__name__)

# No initial neighbors specified here; it will be dynamically adjusted
nn_model = NearestNeighbors(algorithm='auto').fit(df[['x', 'y']])

@app.callback(
    Output('umap-plot', 'figure'),
    [Input('umap-plot', 'clickData')]
)
def display_nearest_neighbors(clickData):
    traces = []
    for label in df['label'].unique():
        df_filtered = df[df['label'] == label]
        traces.append(go.Scatter(
            x=df_filtered['x'],
            y=df_filtered['y'],
            mode='markers',
            marker=dict(size=10),
            name=f'Class {label}',
            customdata=df_filtered.index
        ))

    fig = go.Figure(data=traces)

    if clickData:
        point_index = clickData['points'][0]['customdata']
        selected_point_df = pd.DataFrame([df.iloc[point_index][['x', 'y']].to_numpy()], columns=['x', 'y'])  # Use a DataFrame with column names

        # Iteratively search for the nearest neighbor of the same class
        max_neighbors = len(df)
        for n_neighbors in range(2, max_neighbors + 1):  # Start from 2 to exclude the point itself
            nn_model.set_params(n_neighbors=n_neighbors)
            distances, indices = nn_model.kneighbors(selected_point_df)

            same_class_indices = [i for i in indices[0] if df.iloc[i]['label'] == df.iloc[point_index]['label'] and i != point_index]
            if same_class_indices:
                nearest_same_class_index = same_class_indices[0]
                break

        # Highlight the selected point
        fig.add_trace(go.Scatter(x=[df.iloc[point_index]['x']], y=[df.iloc[point_index]['y']], mode='markers',
                                 marker=dict(color='black', size=15, symbol='circle'),
                                 name='Selected'))

        # Highlight the nearest neighbor overall (which is always the first in the sorted list, after the point itself)
        nearest_overall_index = indices[0][1]  # The first neighbor after the point itself
        nearest_overall = df.iloc[nearest_overall_index]
        fig.add_trace(go.Scatter(x=[nearest_overall['x']], y=[nearest_overall['y']], mode='markers',
                                 marker=dict(color='red', size=12, symbol='x'),
                                 name='Nearest Overall'))

        # Highlight the nearest neighbor of the same class if found
        if same_class_indices:
            nearest_same_class = df.iloc[nearest_same_class_index]
            fig.add_trace(go.Scatter(x=[nearest_same_class['x']], y=[nearest_same_class['y']], mode='markers',
                                     marker=dict(color='blue', size=12, symbol='circle-open'),
                                     name='Nearest Same Class'))

    # Update layout
    fig.update_layout(
        title='Mock UMAP Plot',
        clickmode='event+select',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor='x'),
        width=600,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50),
        legend_title="Class"
    )

    return fig

app.layout = html.Div([
    dcc.Graph(id='umap-plot')
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
