import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go

# Load your data
df_path = 'df.csv'  # Update this path
embedding_df_path = 'embedding_df.csv'  # Update this path
df = pd.read_csv(df_path)
embedding_df = pd.read_csv(embedding_df_path)
df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime is in the correct format

window_size = 96

# Initial setup for the time series plot
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=df['datetime'], y=df['accel_x'], mode='lines', name='Accel X'))
fig_ts.add_trace(go.Scatter(x=df['datetime'], y=df['accel_y'], mode='lines', name='Accel Y'))
fig_ts.add_trace(go.Scatter(x=df['datetime'], y=df['accel_z'], mode='lines', name='Accel Z'))
fig_ts.update_layout(title='Time Series Data')

# Initial setup for the UMAP plot
fig_umap = go.Figure()
fig_umap.add_trace(go.Scatter(x=embedding_df['x'], y=embedding_df['y'], mode='markers', name='UMAP Points'))
fig_umap.update_layout(title='UMAP Embedding')

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='time-series-plot', figure=fig_ts),
    dcc.Graph(id='umap-plot', figure=fig_umap),
])

@app.callback(
    Output('umap-plot', 'figure'),
    [Input('time-series-plot', 'clickData')]
)
def update_umap_from_ts(clickData):
    if clickData:
        # Extract clicked point's datetime from clickData
        clicked_datetime = clickData['points'][0]['x']
        clicked_datetime = pd.to_datetime(clicked_datetime).round('S')  # Round to nearest second
        
        # Ensure DataFrame datetimes are at the same precision level
        # Note: This step is better done once during data loading/preprocessing
        df['datetime'] = pd.to_datetime(df['datetime']).dt.round('S')

        # Find the index of the clicked point in the df
        matching_indices = df.index[df['datetime'] == clicked_datetime]
        if not matching_indices.empty:
            clicked_index = matching_indices[0]
            
            # Calculate which window the clicked index falls into
            window_start_index = (clicked_index // window_size) * window_size
            window_end_index = window_start_index + window_size - 1
            
            # Find the corresponding UMAP point index
            umap_point_index = window_start_index // window_size
            
            # Generate a new UMAP plot highlighting the corresponding point
            fig_umap_updated = go.Figure()
            fig_umap_updated.add_trace(go.Scatter(x=embedding_df['x'], y=embedding_df['y'], mode='markers', name='UMAP Points'))
            # Highlight the corresponding UMAP point
            if umap_point_index < len(embedding_df):
                fig_umap_updated.add_trace(go.Scatter(
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
            fig_umap_updated.update_layout(title='UMAP Embedding')
            
            print("Clicked datetime on raw time-series:", clicked_datetime)
            print("Embedding index of the corresponding UMAP point:", umap_point_index)

            return fig_umap_updated
        else:
            print("Clicked datetime:", clicked_datetime)
            # If no matching datetime is found, log an error or handle gracefully
            print("No matching datetime found in the dataset.")
            return fig_umap  # Return the original UMAP plot if no match is found
    else:
        # Return the original UMAP plot if no point is clicked
        return fig_umap

if __name__ == '__main__':
    app.run_server(debug=True)
