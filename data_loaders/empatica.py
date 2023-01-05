#this code needs to be reworked
#it is not necessary for the functioning of the app
#there are aspects of this code that function perfectly well, but the intention needs to be more fleshed out
#credit goes to Lee Hinkle and the other kind gentlemen at the IMICS lab. 
import pandas as pd
import zipfile
import os
from pathlib import Path
import shutil
class Empatica():
    #have each empatica object keep track of which sensors are going to be incorporated upon instantiation
    
    def __init__(self, subject):        
        self.hr = True
        self.acc = True
        self.bvp = True
        self.eda = True
        self.temp = True
        self.columnLabels = []
        self.mainFrame = pd.DataFrame()
        self.subject = subject
        self.frequencies = []
        self.sensorList = []
        self.binarySensorTrack = [0,0,0,0,0]
        self.freqDict = {"accel" : 32, "bv_pulse" : 64, "electro_dermal" : 4, 
                     "heart_rate" : 1, "temp" : 4}
        self.sensorNames = ["accel", "bvp","eda","hr","temp"]

    def genColumnLabels(self):
        #This will generate the labels corresponding to the sensors that are included
        #in the dataframe, as well as deciding which frequency to use
        if self.acc == True:
            self.columnLabels.extend(["accel_x", "accel_y", "accel_z"])
            self.frequencies.append(32)
            self.binarySensorTrack[0] = 1
        if self.bvp == True:
            self.columnLabels.append("bv_pulse")
            self.frequencies.append(64)
            self.binarySensorTrack[1] = 1
        if self.eda == True:
            self.columnLabels.append("electro_dermal")
            self.frequencies.append(4)
            self.binarySensorTrack[2] = 1
        if self.hr == True:
            self.columnLabels.append("heart_rate")
            self.frequencies.append(1)
            self.binarySensorTrack[3] = 1
        if self.temp == True:
            self.columnLabels.append("temp")
            self.frequencies.append(4)
            self.binarySensorTrack[4] = 1
        
    def processFile(self, ffname):
        """!!!this code needs to be rewritten to incorporate the target 
        frequency into into its time calculation"""
        #all credit goes to Lee Hinkle for this code        
        df = pd.read_csv(ffname, header=None)
        start_time = df.iloc[0,0] # first line in e4 csv
        sample_freq = df.iloc[1,0] # second line in e4 csv
        df = df.drop(df.index[[0,1]]) # drop 1st two rows, index is now off by 2
        df['UTC_time'] = (df.index-2)/sample_freq + start_time
        end_time = df['UTC_time'].iloc[-1]
        df['datetime'] = pd.to_datetime(df['UTC_time'], unit='s')
        df.set_index('datetime',inplace=True)
        df = df.drop('UTC_time', axis=1)
        return df

    def process_accel(self):
        """converts component accel into g and adds accel_ttl column
        per info.txt range is [-2g, 2g] and unit in this file is 1/64g.
        """
        self.mainFrame['accel_x'] = self.mainFrame['accel_x']/64
        self.mainFrame['accel_y'] = self.mainFrame['accel_y']/64
        self.mainFrame['accel_z'] = self.mainFrame['accel_z']/64
        df_sqd = self.mainFrame.pow(2)[['accel_x', 'accel_y', 'accel_z']] # square each accel
        df_sum = df_sqd.sum(axis=1) # add sum of squares, new 1 col df
        self.mainFrame.loc[:,'accel_ttl'] = df_sum.pow(0.5)-1
        fourth_column = self.mainFrame.pop('accel_ttl')
        self.mainFrame.insert(3, 'accel_ttl', fourth_column)
        del df_sqd, df_sum
  
    def setMainFrame(self, df):
        self.mainFrame = df
        
    def uploadMainFrame(self,df):
        self.mainFrame = df
        self.mainFrame.index = pd.to_datetime(self.mainFrame.index)
    
    def getMainFrame(self):
        return self.mainFrame

    def updateMainFrame(self, df):
        self.mainFrame = pd.merge( self.mainFrame, df, how = 'outer'  ,on='datetime')

    def setFinalLabels(self):
        self.genColumnLabels()
        self.mainFrame.columns = self.columnLabels
        self.mainFrame['group'] = self.subject

    def finalizeFrame(self):
        #creates the final labels for the frame and
        #need to add activity labels  
        self.setFinalLabels()
        self.mainFrame = self.mainFrame.interpolate(axis=0)
        self.mainFrame.dropna()

    def fullAssemble(self, path):
        #save the new directory path
        newDirectory = unzipEmpatica(path)

        #change our working directory into the unzipped folder
        os.chdir(newDirectory)
        
        #need to initialize the mainframe with a sensor file
        self.setMainFrame(self.processFile(os.path.join(newDirectory, 'ACC.csv')))

        #keep adding new files to the dataframe by joining paths with file names 
        self.updateMainFrame(self.processFile(os.path.join(newDirectory, 'BVP.csv')))
        self.updateMainFrame(self.processFile(os.path.join(newDirectory, 'EDA.csv')))
        self.updateMainFrame(self.processFile(os.path.join(newDirectory, 'HR.csv')))
        self.updateMainFrame(self.processFile(os.path.join(newDirectory, 'TEMP.csv')))
        self.finalizeFrame()

#this code needs to be rewritten so as to be more general for other OS path formats (i.e. MAC/LINUX)
def unzipEmpatica(path):
    """this first line will change the working directory 
    to the location of the current file """
    #https://note.nkmk.me/en/python-os-getcwd-chdir/
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #this line will need to be adjusted because it norms it for windows 
    fileName = os.path.basename(os.path.normpath(path))

    #unzip the contents of the zipfile and places in directory called temp_ "filename"
    with zipfile.ZipFile(path, 'r') as zipObj:
        zipObj.extractall('temp_' + fileName)
    
    #return the path into the new directory
    return os.path.join(os.getcwd(),'temp_'+fileName)