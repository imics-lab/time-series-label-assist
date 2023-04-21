#all modules that Lee Hinkle used in his code. Included to cover all bases 
import time
from time import gmtime, strftime, localtime #for displaying Linux UTC timestamps in hh:mm:ss
from datetime import datetime
from datetime import timedelta
import numpy as np
#imports for computing and displaying output metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class TimeSeriesNP():
    def __init__(self, time_steps, step):
        self.step = step
        self.time_steps = time_steps

    def timeSlice(self, df):

        #I am doing it this way because in the future the user should be able to 
        #enter a data set that does not contain subject and label columns without
        #breaking the functionality
        N_FEATURES = len(df.columns)
        not_features = 0
        if 'label' in df.columns:
            not_features+=1
        if 'sub' in df.columns:
            not_features+=1
        N_FEATURES-= not_features  

        #subtract 2 for labels and subject columns
        #use seperate arrays for each set of values
        segments = []
        labels = []
        subject = []
        times = []
        steps = 0
        relevantColumns = df.columns[:-not_features]

        #step through the data set and only take 
        for i in range(0, len(df) - self.time_steps, self.step):
            df_segX = df[relevantColumns].iloc[i: i + self.time_steps]
            df_lbl = df['label'].iloc[i: i + self.time_steps]
            df_sub = df['sub'].iloc[i: i + self.time_steps]
            
            ## Save only if labels are the same for the entire segment and valid
            #if (df_lbl.value_counts().iloc[0] != self.time_steps):
            #    continue
            #if 'Undefined' in df_lbl.values :
            #    continue

            subject.append(df['sub'].iloc[i])
            labels.append(df['label'].iloc[i]) 
            segments.append(df_segX.to_numpy())   
            times.append([df.index[i], df.index[i + self.time_steps]])
                

        # Bring the segments into a better shape, convert to nparrays
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, self.time_steps, N_FEATURES)
        labels = np.asarray(labels)
        subject = np.asarray(subject)
        times = np.asarray(times)
        # both labels and sub are row arrays, change to single column arrays
        labels = labels[np.newaxis].T
        subject = subject[np.newaxis].T
        # check for nan - issue with resampled data
        bad_data_locations = np.argwhere(np.isnan(reshaped_segments))
        np.unique(bad_data_locations[:,0]) #[:,0] accesses just 1st column
        if (bad_data_locations.size!=0):
            print("Warning: Output arrays contain NaN entries")
        return reshaped_segments, labels, subject, times
    

    def setArrays(self, df, labels, encode = True, one_hot_encode= True):
        self.x, self.y, self.sub, self.time = self.timeSlice(df)

        #encode using integer or one-hot endcoding
        if encode:
            # y_vector = np.ravel(self.y) #encoder won't take column vector
            # le = LabelEncoder()
            # integer_encoded = le.fit_transform(labels)
            # le.classes_ = labels # added for mapping issue
            # self.mapping = dict(zip(range(len(le.classes_)), le.classes_))
            y_vector = np.ravel(self.y) #encoder won't take column vector
            le = LabelEncoder()

            le.fit(labels)
            le.classes_ = labels
            self.mapping = dict(zip(range(len(le.classes_)), le.classes_))

            integer_encoded = le.fit_transform(y_vector)
            
            if (one_hot_encode):
                onehot_encoder = OneHotEncoder(sparse=False)
                integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                self.y=onehot_encoded
            else:
                self.y = integer_encoded
    

    def updateArrays(self, df):

        #take existing arrays and simply stack the newer ones on the bottom
        temp_x, temp_y, temp_sub, temp_time = timeslice(df)
        self.x = np.vstack([self.x, temp_x])
        self.y = np.vstack([self.y, temp_y])
        self.sub = np.vstack([self.sub, temp_sub])
        self.time = np.vstack([self.time, temp_time])
    
    def finalize(self):

        #final step used in Lee Hinkle's implementation
        #not entirely sure if it necessary
        self.x = np.delete(self.x, (0), axis=0) 
        self.y = np.delete(self.y, (0), axis=0) 
        self.sub = np.delete(self.sub, (0), axis=0)
        self.sub = sub.astype(int) # convert from float to int
    

    """
    use a dictionary to pass in the splitting of the subjects.
    should be useful for experiments but requires more testing
    to ensure proper functionality.
    
    """
    def trainTestSplit_subj(self,
        verbose = True,
        incl_xyz_accel = False, # include component accel_x/y/z in ____X data
        incl_rms_accel = True, # add rms value (total accel) of accel_x/y/z in ____X data
        incl_val_group = False, # split train into train and validate
        split_subj = dict
                    (train_subj = [1,2],
                    validation_subj = [11,21], 
                    test_subj = [3]) 
        ):

        sub_num = np.ravel(self.sub[ : , 0] ) # convert shape to (1047,)
        print(sub_num)
        if (not incl_val_group):
            train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj'] + 
                                            split_subj['validation_subj']))
            x_train = self.x[train_index]
            y_train = self.y[train_index]
        else:
            train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj']))           
            x_train = self.x[train_index]
            y_train = self.y[train_index]

            validation_index = np.nonzero(np.isin(sub_num, split_subj['validation_subj']))
            x_validation = self.x[validation_index]
            y_validation = self.y[validation_index]

        test_index = np.nonzero(np.isin(sub_num, split_subj['test_subj']))
        x_test = self.x[test_index]
        y_test = self.y[test_index]
        if (incl_val_group):
            return x_train, y_train, x_validation, y_validation, x_test, y_test
        else:
            return x_train, y_train, x_test, y_test


            if(verbose):
                headers = ("Reshaped data","shape", "object type", "data type")
                mydata = [("x_train:", x_train.shape, type(x_train), x_train.dtype),
                        ("y_train:", y_train.shape ,type(y_train), y_train.dtype),
                        ("x_test:", x_test.shape, type(x_test), x_test.dtype),
                        ("y_test:", y_test.shape ,type(y_test), y_test.dtype)]
                print(tabulate(mydata, headers=headers))

            return x_train, y_train, x_test, y_test

    def trainTestSplit(self, validation, testSize):
        x_train, x_test, y_train, y_test= sklearn.train_test_split(self.x, self.y, test_size = testSize, random_state = 1)

        if validation:
            x_train, x_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
            return x_train, y_train, x_validation, y_validation, x_test, y_test
        else:
            return x_test, x_train, y_test, y_train
