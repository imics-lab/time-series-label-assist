# Assisted Labeling Visualizer (ALVI): A Semi-Automatic Labeling System for Time-Series Data
A Python-based, multi-page Dash application that provides a comprehensive framework for the semi-automatic labeling of time-series data. It uses semi-supervised learning and visualizations to assist humans in labeling time-series data.

### Preprocessing
  - Application takes in .csv data where the first column is time series data (currently limited to needing the name “datetime”). Following the time series data column, we have sensor data columns, we want a label and subject column, either in the original data, or we add these columns.
<img width="996" alt="image" src="https://github.com/imics-lab/time-series-label-assist/blob/945a3e607655307ed814eae480ff10298b34dd4c/documentation/images/upload_file_format.jpg">

### Imports and Installations

Follow these steps to set up the `alvi` environment with the required libraries using the `requirements.txt` file located at the top of the repository:

1. **Create the conda environment** by running the following command in your terminal:

    ```bash
    conda create --name alvi --file requirements.txt
    ```

2. **Activate the conda environment** by running:

    ```bash
    conda activate alvi
    ```

3. **Run the Dash application** by executing:

    ```bash
    python index.py
    ```

This will start the Dash application using the settings and libraries specified in the `requirements.txt` file installed in the `alvi` environment.

### Pipeline of Application
<img width="996" alt="image" src="https://user-images.githubusercontent.com/108648654/229269330-87851963-0b7e-4631-a535-61dc92a12858.png">

### Data Loading (Files/Video) (Page: Preprocessing)
- **File Uploads**: Users can upload time-series data in CSV format. The expected format is `<timestamp>,<ch1>,...,<chn>,<label>,<sub>`.
- **Video Synchronization**: Accompanying videos can be uploaded and synchronized with the data using specified offsets to enhance labeling accuracy.

### Manual Labeling Interface (Seed Labeling) (Page: Manual Labeling)
- **Interactive Labeling**: Utilize Plotly-based interactive graphs to manually label data. Click on a graph segment to edit or assign new labels.
- **Video Sync**: Synchronize video playback with data points to verify or adjust labels based on visual activities.

### Model Training and Automatic Labeling (Pages: Model Training and Prediction)
- **Model Selection**: Choose between training a new model or using a pre-trained model to label new data automatically.
- **Prediction Interface**: Automatically label new datasets and adjust the model settings as required.

### Data Visualization (Page: Correct Autolabels)
- **UMAP Visualization**: Use UMAP plots to visually inspect the distribution and grouping of labeled data, facilitating easy identification of potential labeling errors.
- **Interactive Corrections**: Click on UMAP or time-series plot points to view and adjust detailed data directly.
- **Synchronization Between Visuals**: Enhance labeling accuracy by synchronizing video playback with UMAP and time-series data. This synchronization allows for simultaneous inspection across all interfaces, facilitating a unified view of each label window prediction.

### Label Correction (Page: Correct Autolabels)
- **Flagged Data Review**: Review and correct labels flagged by the system based on confidence thresholds set by the user.
- **Comprehensive Interface**: Combine video, raw time series, and UMAP visualizations for a thorough review process.

### Finalizing and Exporting Data
- **Data Integration**: Post-correction, integrate the refined labels back into the model to improve its predictions.
- **Export**: Export the correctly labeled data for further use or analysis.

