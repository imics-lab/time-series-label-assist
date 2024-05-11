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

### Repository Structure

This repository is structured to support the functionality of the Assisted Labeling Visualizer (ALVI), a semi-automatic labeling system for time-series data. Below is an outline of the key components and their roles:

```
.
├── apps                           # All individual Dash apps that give functionality to the application
│   ├── preprocessing.py           # Handles the uploading and initial processing of time-series data
│   ├── manual_labeling.py         # Provides the interface for manual data labeling using interactive graphs
│   ├── model_training.py          # Manages training of new models or utilization of pre-trained models
│   ├── prediction.py              # Facilitates the prediction process using trained models
│   └── correct_autolabels.py      # Enables review and correction of labels suggested by the automatic labeling process
│
├── assets                         # Stores videos temporarily for rendering in Dash app, deleted after labeling
│
├── data                           # Prepopulated with datasets tested on the application
│   ├── TWristAR                   # HAR dataset with structured and unstructured activities, includes full video records
│   └── CMU-MMAC                   # Multimodal dataset from various kitchen activities, supports complex environment testing
│
├── data_loaders                   # Helper scripts to load datasets
│
├── documentation                  # Comprehensive documentation of the app and its development
│
├── labeled_data                   # Default directory for saving labeled data, can be set to another directory by user
│
├── prediction                     # Stores models, model dimensions, and classification details used in training and prediction
│
├── storage                        # Temporary live storage used during application operation, cleared after closing the app
│
├── app.py                         # Initializes the Dash app, sets up server and configurations
│
└── index.py                       # Manages routing and interaction of different app pages, implements core app logic
```

## Key Files Explained
- **apps directory**: Contains the individual Dash applications that provide the core functionality of our app in different pages.
- **index.py**: This is the main entry point of the Dash application. It sets up the web server, defines the navigation between different pages of the app, and manages user access to these pages based on configuration settings.

## Video Walkthrough
[https://youtu.be/rYPZxyt82KI](https://youtu.be/rYPZxyt82KI)

[![Video thumbnail](https://img.youtube.com/vi/rYPZxyt82KI/0.jpg)](https://www.youtube.com/watch?v=rYPZxyt82KI)
