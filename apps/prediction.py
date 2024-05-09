from dash import html
import dash_bootstrap_components as dbc

# preprocessing + data upload
# show predictions
# load necessary data for next interface into assets
import base64
import io
import os
import shutil
import pandas as pd
from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from prediction import model, split
import json
import pickle

def layout():
    models_path = os.path.join("prediction", "models")
    if os.path.isdir(models_path):
        dir_list = [model for model in os.listdir(models_path) if model.endswith('.h5')]
        if dir_list:
            model_select_component = dcc.RadioItems(
                options=[{'label': model, 'value': model} for model in dir_list],
                value=dir_list[0],
                id='model-select'
            )
        else:
            model_select_component = html.Div("No models found.")
    else:
        model_select_component = html.Div("Model directory not found.")

    return html.Div([
        html.H3('Upload and Predict Interface'),
        html.H5('Select or Upload Model'),
        dbc.Row([
            dbc.Col(model_select_component, width=12),
            dbc.Col(dbc.Button('Submit Selection', id='submit-model', n_clicks=0, className='btn-primary'), width=12)
        ]),
        html.Div(id='model-submit-output'),

        html.H5('Upload dataset you wish to have labeled'),
        dcc.Upload(
            id='upload-data',
            children=dbc.Button('Upload Data', className='btn-secondary'),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='dataset-load-output'),

        html.H5('Upload video data'),
        dcc.Upload(
            id='upload-video',
            children=dbc.Button('Upload Video', className='btn-secondary'),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='video-load-output'),
        html.H5("Video Data Sync Offset", style={'marginBottom': '10px'}),
        html.Div([
            html.P([html.Strong("Zero Offset:"), " Sync Start - Video and data begin together."]),
            html.P([html.Strong("Positive Offset:"), " Video Delay - Start video [offset] seconds after data."]),
            html.P([html.Strong("Negative Offset:"), " Data Delay - Start data [offset] seconds after video."]),
            dbc.Input(id='video-data-offset', type='number', placeholder="Enter offset in seconds...", step=1, style={'marginBottom': '10px'}),
        ], style={'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col(html.Div([
                html.Label('Enter Window Size:'),
                dbc.Input(id='window-size-pred', type='number', value=96, placeholder='Window Size', className='mb-2')
            ]), width=4),
            dbc.Col(html.Div([
                html.Label('Enter Steps:'),
                dbc.Input(id='steps-pred', type='number', value=96, placeholder='Steps', className='mb-2')
            ]), width=4),
            dbc.Col(html.Div([
                html.Label('Enter Confidence Threshold:'),
                dbc.Input(id='confidence-threshold', type='number', value=0.90, placeholder='Confidence Threshold', className='mb-2')
            ]), width=4)
        ]),
        dbc.Button('Predict Labels', id='predict-labels-button', n_clicks=0, className='btn-success mt-3'),
        html.Div(id='prediction-output')
    ])

def labelDf(df, labels_dict, model, npObject):
    #copy the dataframe and initialize all predicted labels to Undefined
    labeledDf = df.copy()
    labeledDf['pred_labels'] = "Other"

    #reformat y_pred to hold the string values for labels
    predictions = model.predict(npObject.x, verbose = 0, batch_size = 32)
    y_pred = np.argmax(predictions, axis=-1)
    y_pred_labels = np.vectorize(labels_dict.get)(y_pred)
    unique, counts = np.unique(y_pred_labels, return_counts=True)

    #print out relevant information about labels and dictionary
    print("Final Predicted Label Counts: ")
    print (np.asarray((unique, counts)).T)
    #label the new dataframe

    for z in range(y_pred_labels.size):
        start = npObject.time[z][0]
        end = npObject.time[z][1]
        labeledDf.loc[start : end, ['pred_labels']] = y_pred_labels[z]

    return predictions, labeledDf

def update_config(confidence_threshold, window_step_size, video_data_offset):
    config_path = os.path.join('', 'config.json')
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        config = {}

    config['window-and-step-size'] = window_step_size
    config['conf_thresh'] = confidence_threshold
    config['can_access_correct_autolabels'] = True
    config['offset_pred'] = video_data_offset
    
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

@callback(
    Output('model-submit-output', 'children'),
    Input('submit-model', 'n_clicks'),
    State('model-select', 'value'),
    prevent_initial_call=True
)
def submit_model_selection(n_clicks, selected_model):
    if n_clicks > 0:
        selected_model_path = os.path.join("prediction", "models", selected_model)
        assets_dir = "storage"
        if os.path.isfile(selected_model_path) and os.path.isdir(assets_dir):
            shutil.copy(selected_model_path, assets_dir)
            return f"Model copied to storage directory."
        else:
            return "Model not found or storage directory does not exist."

@callback(
    Output('dataset-load-output', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def load_dataset(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                save_path = os.path.join("storage", "auto_label_df.csv")
                df.to_csv(save_path, index=False)
                return f"Dataset loaded and saved to {save_path}"
            else:
                return "Unsupported file type"
        except Exception as e:
            return f"Error processing file: {e}"
    else:
        raise PreventUpdate


@callback(
    Output('video-load-output', 'children'),
    Input('upload-video', 'contents'),
    State('upload-video', 'filename'),
    prevent_initial_call=True
)
def load_video(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        assets_dir = "assets"
        output_file_name = "autolabel_video.mp4"
        save_path = os.path.join(assets_dir, output_file_name)
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)
        with open(save_path, 'wb') as f:
            f.write(decoded)
        return f"Video copied to {save_path}"
    else:
        raise PreventUpdate

@callback(
    Output('prediction-output', 'children'),
    Input('predict-labels-button', 'n_clicks'),
    State('window-size-pred', 'value'),
    State('steps-pred', 'value'),
    State('confidence-threshold', 'value'),
    State('video-data-offset', 'value'),  # Add this state parameter
    State('model-select', 'value')
)
def predict_labels(n_clicks, window_size, steps, confidence_threshold, video_data_offset, model_name):
    if n_clicks > 0:
        # Update the configuration file with the new confidence threshold
        update_config(confidence_threshold, window_size, video_data_offset)

        # Load the model
        model_path = os.path.join("prediction", "models", model_name)
        model_loaded = load_model(model_path)

        # Load and prepare the data
        data_path = os.path.join("storage", "auto_label_df.csv")
        df = pd.read_csv(data_path)
        label_list_path = os.path.join("storage", "label_list.csv")
        labelList = pd.read_csv(label_list_path)
        labelList = list(labelList)

        if 'sub' in df.columns:
            df['sub'] = "Undefined"
        df = df.set_index('datetime')

        np_labeling = split.TimeSeriesNP(window_size, steps)
        print("Label list used:\n", labelList)
        np_labeling.setArrays(df, encode=True, one_hot_encode=False, labels=labelList, filter=False)
        
        testModel = model.CNN()
        testModel.setModel(model_loaded)
        testModel.only_test_data(np_labeling.x, np_labeling.y)

        print("Label Mapping: ", np_labeling.mapping)
        predictions, labeledDf = labelDf(df, np_labeling.mapping, testModel.model, np_labeling)

        # Identify the position of the original 'label' column, if it exists
        if 'label' in labeledDf.columns:
            # Get the index of the original 'label' column
            label_index = labeledDf.columns.get_loc('label')
            # Drop the original 'label' column
            labeledDf = labeledDf.drop(columns=['label'])
            # Insert 'pred_labels' column at the position of the original 'label' column
            labeledDf.insert(label_index, 'label', labeledDf['pred_labels'])
            # Drop the 'pred_labels' as it's now renamed and moved
            labeledDf = labeledDf.drop(columns=['pred_labels'])
        else:
            # If no original 'label' column, just rename 'pred_labels' to 'label'
            labeledDf = labeledDf.rename(columns={'pred_labels': 'label'})

        np_labeling.setArrays(labeledDf, encode=True, one_hot_encode=False, labels=labelList, filter=False)

        # Define a path to the assets directory
        prediction_directory = 'storage'
        if not os.path.exists(prediction_directory):
            os.makedirs(prediction_directory)

        file_path = os.path.join(prediction_directory, 'np_auto_labeling.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(np_labeling, f)   

        labeled_file_path = os.path.join('', "storage", 'auto_label_df.csv')
        # Check if there's a datetime column in your DataFrame
        if 'datetime' in labeledDf.columns:
            labeledDf.to_csv(labeled_file_path, index=False)
        elif 'datetime' in df.index:
            labeledDf.reset_index(inplace=True)
            labeledDf.to_csv(labeled_file_path, index=False)
        print(f"Updated dataset saved to {labeled_file_path}")

        labeled_array_file_path = os.path.join('', "storage", "predictions.npy")
        np.save(labeled_array_file_path, predictions)
        print(f"NumPy predictions array saved to {labeled_array_file_path}")        

        # Extract label counts from the updated dataframe
        label_counts = labeledDf['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        label_counts_html = [
            html.Tr([html.Td(label), html.Td(str(count))]) for label, count in label_counts.itertuples(index=False)
        ]

        # Save updated DataFrame
        labeled_file_path = os.path.join("storage", 'auto_label_df.csv')
        labeledDf.to_csv(labeled_file_path, index=True)
        print(f"Updated dataset saved to {labeled_file_path}")

        # Load existing configuration from file
        config_path = 'config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = json.load(file)
        else:
            config = {}

        # Update configuration with new values
        config.update({
            "can_access_correct_autolabels": True  # 
        })

        # Save the updated configuration back to the JSON file
        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)

        return html.Div([
            html.H5("Label Mapping:"),
            html.Pre(str(np_labeling.mapping)),
            html.H5("Final Predicted Label Counts on DF Data:"),
            html.Table(
                [html.Tr([html.Th("Label"), html.Th("Count")])] + label_counts_html
            )
        ])
    raise PreventUpdate