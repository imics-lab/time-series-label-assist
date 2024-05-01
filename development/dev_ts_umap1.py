# Required libraries
import pandas as pd
import plotly.graph_objects as go

def load_data():
    """
    Load the time series and UMAP embedding data from CSV files.
    """
    df_path = 'df.csv'  # Adjust path as necessary
    embedding_df_path = 'embedding_df.csv'  # Adjust path as necessary

    df = pd.read_csv(df_path)
    embedding_df = pd.read_csv(embedding_df_path)

    # Convert 'datetime' to datetime object for easier manipulation
    df['datetime'] = pd.to_datetime(df['datetime'])

    return df, embedding_df

def define_segment(df, window_size=96):
    """
    Define segments of the time series data based on the sliding window size.
    For simplicity, this example uses the first window to define a single segment.
    """
    # Assuming the start of the first window
    start_index = 0
    end_index = start_index + window_size - 1  # Adjust for zero-based indexing
    
    # Return the segment and the indices of the segment
    return df.iloc[start_index:end_index + 1], start_index, end_index

def map_to_umap(segment_start_index, window_size, embedding_df):
    """
    Map the segment defined by its start index in the time series to the corresponding UMAP point.
    Assumes each UMAP point corresponds to a sliding window of the specified size in the time series.
    """
    # Calculate the UMAP point index based on the segment's start index
    # This calculation assumes non-overlapping windows; adjust if your windows overlap
    umap_point_index = segment_start_index // window_size
    
    # Return the corresponding UMAP point(s)
    return embedding_df[embedding_df['index'] == umap_point_index]

def visualize_data(df, segment, umap_points, embedding_df):
    """
    Create visualizations for the time series segment and corresponding UMAP points with highlighting.
    """
    # Time Series Plot
    fig_ts = go.Figure()

    # Add traces for full time series data
    fig_ts.add_trace(go.Scatter(x=df['datetime'], y=df['accel_x'], mode='lines', name='Accel X'))
    fig_ts.add_trace(go.Scatter(x=df['datetime'], y=df['accel_y'], mode='lines', name='Accel Y'))
    fig_ts.add_trace(go.Scatter(x=df['datetime'], y=df['accel_z'], mode='lines', name='Accel Z'))

    # Highlight the selected segment
    fig_ts.add_trace(go.Scatter(x=segment['datetime'], y=segment['accel_x'], mode='markers', name='Highlighted Accel X', marker=dict(color='red')))
    fig_ts.add_trace(go.Scatter(x=segment['datetime'], y=segment['accel_y'], mode='markers', name='Highlighted Accel Y', marker=dict(color='red')))
    fig_ts.add_trace(go.Scatter(x=segment['datetime'], y=segment['accel_z'], mode='markers', name='Highlighted Accel Z', marker=dict(color='red')))
    
    fig_ts.update_layout(title='Time Series Data with Highlighted Segment')

    # UMAP Plot
    fig_umap = go.Figure()

    # Add trace for UMAP points
    fig_umap.add_trace(go.Scatter(x=embedding_df['x'], y=embedding_df['y'], mode='markers', marker=dict(size=5), name='UMAP Points'))

    # Highlight corresponding UMAP points
    fig_umap.add_trace(go.Scatter(x=umap_points['x'], y=umap_points['y'], mode='markers', marker=dict(color='red', size=7), name='Highlighted UMAP Points'))
    
    fig_umap.update_layout(title='UMAP Embedding with Highlighted Points')

    # Show plots
    fig_ts.show()
    fig_umap.show()

def main():
    """
    Adjust the main function to execute the workflow with the new mapping logic.
    """
    window_size = 96  # Define the sliding window size used in preprocessing
    df, embedding_df = load_data()
    segment, segment_start_index, _ = define_segment(df, window_size)
    umap_points = map_to_umap(segment_start_index, window_size, embedding_df)
    
    # Visualize the selected time series segment and corresponding UMAP points with highlighting
    visualize_data(df, segment, umap_points, embedding_df)

if __name__ == "__main__":
    main()
