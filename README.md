# Assisted Labeling Visualizer (Alvi): A Semi-Automatic Labeling System for Time-Series Data
A Python-based labeling tool that uses self-supervised learning and visualizations to assist humans in labeling time series data

### Preprocessing
  - Application takes in .csv data where the first column is time series data (currently limited to needing the name “datetime”). Following the time series data column, we have sensor data columns, we want a label and subject column, either in the original data, or we add these columns.
  
### Accessing application "Open Google Colab notebook"
  - We can open the notebook in Google Colab through GitHub
    1) File -> Open Notebook in Colab
    2) Select GitHub option
    3) Paste link to GitHub Repo (https://github.com/imics-lab/time-series-label-assist)
    4) Click latest master_labeling_interface.ipynb
  - Download notebook and open from file explorer
    1) At GitHub repo (https://github.com/imics-lab/time-series-label-assist)
    2) Locate latest master_labeling_interface.ipynb
    3) File -> Open Notebook in Colab
    4) Select Upload option


### Initial Installs
  - Google Colab has numerous Python packages, but throughout development of this application, we have needed earlier/later versions of some of the main packages, or packages that aren’t already on Colab. These installs are in a cell named “Set up installs and imports” and take around 1 to 2 minutes to install once run in the notebook.
 
### Pipeline of Application
<img width="996" alt="image" src="https://user-images.githubusercontent.com/108648654/229269330-87851963-0b7e-4631-a535-61dc92a12858.png">

### Uploading files/video
  -For data, we use upload widgets to select a data file (our .csv in the format described above in the preprocessing step) and a label list (which is also a .csv, but of all possible labels for the dataset).
  - For video, we have two options, we can upload a video file directly to the Colab session, similar to the data upload (takes a long time if video is not short). Or, we can specify a Google Drive link to the video file.
	- The Google Drive option is better, because the user only needs to upload the video file once to their Google Drive, then they can use that link to repeatedly use the application.


### Initial Labeling Interface
  - The user will manually label their uploaded .csv data, in our specified format. We load the data with functionality to manually add labels. To help aid the user in manually adding labels, we have interactive graphing callbacks (click on graph and fill UI), and a way to sync the video to the data.

### Model training
  - Once we manually label the data, we have two options. We can either build and train a model using that data, save that model, then load it to generate predictions on unlabeled data within the same dataset. Or, we can have a pretrained model, and bypass the build/save model, and generate predictions that way. 
  - We then have the option to choose a model to load, and upload a new .csv (in our format), that we then generate predictions for.
  - Currently use the TWristAR dataset with a pretrained model for testing of the application.


### Label Visualization (UMAP)
  - We utilize UMAP to visualize the manually labeled data, providing users with an additional perspective alongside the video. Through this representation of the data features, the user can correct label inaccuracies.
  - On top of visualizing the UMAP, we have the ability to click on a point, view its nearest neighbor of the same type and overall, visualize each points segment on the raw time series graph, and then the ability to modify the label for that point on the UMAP.


### Label Correction
  -This interface is our biggest, and combines features of all the previous interfaces. We load the unlabeled uploaded data we have generated predictions for. We display the video associated (if we have it), the raw time series of the data, and the UMAP. Alongside these representations of the data, we have all the same functionality as previously talked about.
  - The user will specify a confidence threshold in a UI before this cell is run. If the confidence of the model's prediction is below this threshold, there will be a flag on the raw time series for the user to review in this stage. The user will utilize all these representations to correct the flagged labels.


### Back to Model Training
  - Following the label correction process, we need to integrate the newly labeled data into the model to enhance its predictive capabilities.

### Eventually Leave App with Newly Labeled Data
  - save labeled data

