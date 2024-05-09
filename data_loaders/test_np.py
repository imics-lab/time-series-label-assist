import os
import pickle
from prediction import model, split
import pandas as pd
import numpy as np
import tensorflow as tf

# Load and prepare the data
data_path = os.path.join("assets", "auto_label_df.csv")
df = pd.read_csv(data_path)
df['sub'] = "Undefined"
df = df.set_index('datetime')
label_list_path = os.path.join("assets", "label_list.csv")
labelList = pd.read_csv(label_list_path)
labelList = list(labelList)
model_loaded = tf.keras.models.load_model('prediction/models/TWristARmodel.h5')

np_labeling = split.TimeSeriesNP(96, 96)
print("Label list used:\n", labelList)
np_labeling.setArrays(df, encode=True, one_hot_encode=False, labels=labelList, filter=False)

print(np_labeling.y)

testModel = model.CNN()
testModel.setModel(model_loaded)
testModel.only_test_data(np_labeling.x, np_labeling.y)

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

print("Label Mapping: ", np_labeling.mapping)
predictions, labeledDf = labelDf(df, np_labeling.mapping, testModel.model, np_labeling)
labeledDf["label"] = labeledDf["pred_labels"]
labeledDf = labeledDf.drop('pred_labels', axis=1)
np_labeling.setArrays(labeledDf, encode=True, one_hot_encode=False, labels=labelList, filter=True)
print(np_labeling.y)

with open(os.path.join('prediction', 'np_auto_labeling.pkl'), 'rb') as file:
    auto_new_np = pickle.load(file)
print(auto_new_np.y)

# with open(os.path.join('prediction', 'new_np.pkl'), 'rb') as file:
#     manual_new_np = pickle.load(file)
# print(manual_new_np)

# label_num_dict_auto = {auto_new_np.mapping[k] : k for k in auto_new_np.mapping}
# label_num_dict_manual = {manual_new_np.mapping[k] : k for k in manual_new_np.mapping}
# print(label_num_dict_auto)
# print(label_num_dict_manual)

# npInput_auto = auto_new_np.y
# npInput_manual = manual_new_np.y

# print(type(npInput_auto))
# print(type(npInput_manual))

# print(npInput_auto)
# print(npInput_manual)

# # 1st, replace everything that saves to assets OTHER THAN VIDEO. to save to a new folder for easier interface debugging
# # error is now that we are only getting other labels, even though our predictions are not only other.

# # in split.py undefined bug... issue?

# np_labeling = split.TimeSeriesNP(96, 96)
# print("Label list used:\n", labelList)
# np_labeling.setArrays(df, encode=True, one_hot_encode=False, labels=labelList, filter=False)

# testModel = model.CNN()
# testModel.setModel(model_loaded)
# testModel.only_test_data(np_labeling.x, np_labeling.y)

# def labelDf(df, labels_dict, model, npObject):
#     #copy the dataframe and initialize all predicted labels to Undefined
#     labeledDf = df.copy()
#     labeledDf['pred_labels'] = "Other"

#     #reformat y_pred to hold the string values for labels
#     predictions = model.predict(npObject.x, verbose = 0, batch_size = 32)
#     y_pred = np.argmax(predictions, axis=-1)
#     y_pred_labels = np.vectorize(labels_dict.get)(y_pred)
#     unique, counts = np.unique(y_pred_labels, return_counts=True)

#     #print out relevant information about labels and dictionary
#     print("Final Predicted Label Counts: ")
#     print (np.asarray((unique, counts)).T)
#     #label the new dataframe

#     for z in range(y_pred_labels.size):
#         start = npObject.time[z][0]
#         end = npObject.time[z][1]
#         labeledDf.loc[start : end, ['pred_labels']] = y_pred_labels[z]

#     return predictions, labeledDf

# print("Label Mapping: ", np_labeling.mapping)
# predictions, labeledDf = labelDf(df, np_labeling.mapping, testModel.model, np_labeling)

# print(np_labeling.y)
# print(labeledDf.head())

# with open(os.path.join('assets', 'time_series_processed_data.pkl'), 'rb') as file:
#     test = pickle.load(file)
# print(test.y)