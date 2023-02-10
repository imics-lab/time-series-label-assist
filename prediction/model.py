import tensorflow as tf
from tensorflow import keras #added to save model
from tensorflow.keras import layers #format matches MNIST example
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense 
from sklearn.metrics import accuracy_score
import numpy as np
import sys
sys.path.append('..')
from preprocessing import split

#class only works
class CNN():

    #give the user seperate options for whether or not they want to use validation data
    def importNP(self, x_train, y_train, x_test, y_test):
        self.validation = False
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_timesteps = self.x_train.shape[1]
        self.n_features = self.x_train.shape[2]
        self.n_outputs = self.y_train.shape[1]

    def importNPV(self, x_train, y_train, x_validation, y_validation, x_test, y_test):
        self.validation = True
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.n_timesteps = x_train.shape[1]
        self.n_features = x_train.shape[2]
        self.n_outputs = y_train.shape[1]

    #use all of the data to traint he model
    def only_train_data(self, x_train, x_validation, y_train, y_validation ):
        self.validation = True

        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation

        self.n_timesteps = self.x_train.shape[1]
        self.n_features = self.x_train.shape[2]
        if len(y_train.shape) > 1:
            self.n_outputs = self.y_train.shape[1]
        else:
            self.n_outputs = self.y_train.shape[0]
    
    def only_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
    
    def setModel(self, model):
        self.model = model

    def build(self, num_filters = 100, k1_size=17, k2_size=17, dropout = 0.5, mpool_size = 2):
        #need to grab output of dense layer before softmax
        self.model = keras.Sequential(
            [
                layers.Input(shape=self.x_train[0].shape),
                layers.BatchNormalization(scale=False),
                layers.Conv1D(filters=num_filters, kernel_size=k1_size, activation='relu',input_shape=(self.n_timesteps,self.n_features)),
                layers.Dropout(dropout),
                layers.Conv1D(filters=num_filters, kernel_size=k2_size, activation='relu'),
                layers.Dropout(dropout),
                layers.MaxPooling1D(pool_size=mpool_size),
                layers.Flatten(),
                layers.Dense(100, activation='relu'),
                layers.Dense(50, activation='relu'),
                layers.Dense(self.n_outputs, activation='softmax')
                ]) 

    def train(self, BATCH_SIZE = 32, MAX_EPOCHS = 200): # Max number run unless earlystopping callback fires
        # increasing dropout to 0.8 and higher requires more than 100 epochs
        # see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
        callback = EarlyStopping(monitor='val_loss', mode = 'min', patience=20)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        if self.validation:
            self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size = BATCH_SIZE,
            epochs=MAX_EPOCHS,
            callbacks=[callback],
            validation_data=(self.x_validation,self.y_validation),
            verbose = 0) #0 = silent, 1 = progress bar, 2 = one line per epoch
        else:
            self.history = self.model.fit(
                self.x_train,self.y_train,
                batch_size = BATCH_SIZE,
                epochs=MAX_EPOCHS,
                callbacks=[callback],
                verbose = 0) #0 = silent, 1 = progress bar, 2 = one line per epoch

    def run(self, returnOneHot=False):
        self.predictions = self.model.predict(self.x_test, verbose=0,batch_size=32)
        # must use values not one-hot encoding, use argmax to convert
        y_pred = np.argmax(self.predictions, axis=-1) # axis=-1 means last axis
        y_temp = np.argmax(self.y_test, axis=-1)
        if returnOneHot:
            return (accuracy_score(y_temp, y_pred)),y_pred, predictions
        else:
            return (accuracy_score(y_temp, y_pred)),y_pred