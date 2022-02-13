import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os, codecs
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN
import sklearn.preprocessing as preprocessing
import pickle

class FeatureLabelExplorer:
    def __init__(self, dataFrom, dataTo=None):
        self.dataFrom = dataFrom
        self.dataTo = dataTo
    
    def __saveDict(self, dictionary, count, invalidFeatureDict):
        with open(os.path.join(self.dataTo, 'featureDict.pkl'), 'wb') as fwb:
            pickle.dump({"dict": dictionary, "count": count, "invalidFeatureDict": invalidFeatureDict}, fwb)
    
    def __showResults(self, featureDict, newColIndex, invalidFeatureDict):
        print("\nfeatureDict is built, encoded features number: %d" % newColIndex)
        for columnStr in featureDict.keys():
            print("Columns: %s)" % (columnStr))
            for key,value in featureDict[columnStr].items():
                print("\t%s -> %d" % (key, value))
        print("invalidFeatureDict: %s" % invalidFeatureDict)
        print("FeatureDict loaded")
    
    def loadfeatureDict(self):
        dict_ = None
        with open(os.path.join(self.dataFrom, 'featureDict.pkl'), 'rb') as frb:
            dict_ = pickle.load(frb)
        self.__showResults(dict_["dict"], dict_["count"], dict_["invalidFeatureDict"])
        return dict_["dict"], dict_["count"], dict_["invalidFeatureDict"]
        
    #returned featureDict: {columnStr1: dict(value1: newColIndex1, value2: newColIndex2, ...), columnStr2: set(...), ...}
    #returned invalidFeatureDict: {columnStr1: set(value1, value2, ...), ...)
    #some labels can easily be reduced
    def exploreLabel(self, threshold=1000):
        print("FeatureLabelExplorer:")
        featureDict = {}
        excludedfeatureDict=set(["transaction_date", "CASHING_SUM", "SEX_CD", "HOME_PREFECTURE_CD", "WORK_PREF_SAME_AS_HOME_FLG"])
        for filePath in sorted(os.listdir(self.dataFrom)):
            print(filePath)
            with codecs.open(os.path.join(self.dataFrom, filePath), "r", "Shift-JIS", "ignore") as fp:
                df = pd.read_csv(fp)
                for columnStr in df.columns:
                    if columnStr not in excludedfeatureDict:
                        if columnStr not in featureDict.keys():
                            featureDict[columnStr] = {}
                        for i in range(len(df)):
                            value = df[columnStr][i]
                            if not pd.isna(value):
                                value = str(value)
                                featureDict[columnStr][value] = featureDict[columnStr].get(value, 0)+1
        #filt bad labels
        newColIndex = 0
        invalidFeatureDict = {}
        for columnStr in featureDict.keys():
            for key,value in featureDict[columnStr].items():
                if value < threshold:
                    if columnStr not in invalidFeatureDict:
                        invalidFeatureDict[columnStr] = set()
                    invalidFeatureDict[columnStr].add(key)
                else:
                    featureDict[columnStr][key] = newColIndex
                    newColIndex += 1
        for columnStr in invalidFeatureDict.keys():
            for key in invalidFeatureDict[columnStr]:
                del featureDict[columnStr][key]
                print("Invalid feature deleted: %s -> %s" % (columnStr, key))
        #create 3 references to the same empty dict, bad    
        #featureDict["SEX_CD"], featureDict["HOME_PREFECTURE_CD"], featureDict["WORK_PREF_SAME_AS_HOME_FLG"]  = [{}]*3    
        featureDict["SEX_CD"] = {}
        for i in np.arange(1,3):
            featureDict["SEX_CD"][str(float(i))] = newColIndex
            newColIndex += 1
        featureDict["HOME_PREFECTURE_CD"] = {}
        for i in np.arange(1,48):
            featureDict["HOME_PREFECTURE_CD"][str(float(i))] = newColIndex
            newColIndex += 1
        featureDict["WORK_PREF_SAME_AS_HOME_FLG"] = {}
        for i in np.arange(0,2):
            featureDict["WORK_PREF_SAME_AS_HOME_FLG"][str(float(i))] = newColIndex
            newColIndex += 1    
            
        self.__saveDict(featureDict, newColIndex, invalidFeatureDict)        
        self.__showResults(featureDict, newColIndex, invalidFeatureDict)
        return featureDict, newColIndex, invalidFeatureDict               

# get 2D data from DataManager
class DataManager:
    #featureDict: a three elements tuple. [dataMonthStart, dataMonthEnd)
    def __init__(self, dataFrom, dataMonthStart="2020-02", dataMonthEnd="2021-10", isConvertData=False, featureDict=None, dataTo=None, isExcludeInvalidFeature=True):
        # ndarray
        self.featuresData = None
        self.labelsData = None
        self.isExcludeInvalidFeature = isExcludeInvalidFeature
        self.featureDataMonthStartStr = "2018-09"
        #"2021-10" is excluded
        self.featureDataMonthEndStr = "2021-10"
        self.dataMonthStartStr = dataMonthStart
        self.dataMonthEndStr = dataMonthEnd
        print("DataManager")
        if isConvertData:
            featuresDataList = []
            labelsDataList = []
            self.featureDict = featureDict[0]
            self.featureNum = featureDict[1]
            self.invalidFeatureDict = featureDict[2] 
            for filePath in sorted(os.listdir(dataFrom)):
                print(filePath)
                with codecs.open(os.path.join(dataFrom, filePath), "r", "Shift-JIS", "ignore") as fp:
                    df = pd.read_csv(fp)
                    df.loc[:, ["SEX_CD", "HOME_PREFECTURE_CD", "WORK_PREF_SAME_AS_HOME_FLG"]] = df.loc[:, ["SEX_CD", "HOME_PREFECTURE_CD", "WORK_PREF_SAME_AS_HOME_FLG"]].astype("float").round()  
                    validDf = self.__removeNanValue(df)
                    subFeaturesData, subLabelsData = self.__convertDataFrame(validDf, ["transaction_date"])
                    
                    #debug
                    print("Converted subFeaturesData")
                    print(subFeaturesData)
                    print(subFeaturesData.shape)
                    print("Converted subLabelsData")
                    print(subLabelsData)
                    print(subLabelsData.shape)
                    
                    featuresDataList.append(subFeaturesData)
                    labelsDataList.append(subLabelsData)
            self.featuresData = np.concatenate(featuresDataList, axis=0)
            self.labelsData = np.concatenate(labelsDataList, axis=0)
            np.save(os.path.join(dataTo, 'featuresData'), self.featuresData)
            np.save(os.path.join(dataTo, 'labelsData'), self.labelsData)
            print("Data saved")
            #debug
            print("Aggregated featuresData")
            print(self.featuresData)
            print(self.featuresData.shape)
            print("Aggregated labelsData")
            print(self.labelsData)
            print(self.labelsData.shape)
            print("Debugging for DataManager is done")
        else:
            self.featuresData = np.load(os.path.join(dataFrom, "featuresData.npy"))
            self.labelsData = np.load(os.path.join(dataFrom, "labelsData.npy"))
        print("Data loaded")
    
    # some other works may be done here in the future, you may filling the NAN instead of dropping it    
    def __removeNanValue(self, df):
        deletedRows = set()
        if self.isExcludeInvalidFeature:
            for columnStr in self.invalidFeatureDict.keys():
                for row in range(len(df)):
                    value = str(df[columnStr][row])
                    if value in self.invalidFeatureDict[columnStr]:
                        deletedRows.add(row)
            deletedRows = sorted(list(deletedRows))
            for row in deletedRows:
                print("row %d in DataFrame will be deleted." % row)
                print("\t%s" % (df.iloc[row].to_numpy()))
            df.drop(deletedRows)
        return df.dropna()

    def __convertDataFrame(self, df, excludedColumns):
        featureDict = self.featureDict
        # cashingSum for each time_step(day), the date info is dropped
        newDfGb = df.groupby(['transaction_date'])
        cashingSumArray = np.zeros(len(newDfGb.groups))
        featureArray = np.zeros((len(cashingSumArray), self.featureNum))
        dateList = []
        for dateStr, rowList in newDfGb.groups.items():
            dateList.append((dateStr, rowList))
        dateList.sort(key=lambda x : x[0])
        
        for i in range(len(dateList)):
            # in one time step(one day)
            for dfCol in range(len(df.columns)):
                columnStr = df.columns[dfCol]
                if columnStr not in excludedColumns:
                    if columnStr != "CASHING_SUM":
                        for dfRow in range(len(dateList[i][1])):
                            feature = str(df.iloc[dfRow, dfCol])
                            #filt bad features
                            if feature in featureDict[columnStr].keys():
                                featureArray[i, featureDict[columnStr][feature]] += 1
                    else:
                        for dfRow in range(len(dateList[i][1])):
                            cashingSumArray[i] += df.iloc[dfRow, dfCol]
        return featureArray, cashingSumArray
    
    def __countDaysFromMonthPeriod(self, monthStartStr, monthEndStr):
        monthStart = datetime.strptime(monthStartStr, "%Y-%m")
        monthEnd = datetime.strptime(monthEndStr, "%Y-%m")
        return (monthEnd - monthStart).days
    
    def __checkLoadedData(self):
        print("featuresData: %s\nlabelsData: %s" % (self.featuresData.shape, self.labelsData.shape))
        assert self.featuresData.shape[0] == self.labelsData.shape[0], "Invalid loaded data"
    
    def __dataPreprocess(self, featuresData_, labelsData_, labelDiv=5e9):
        featureData = preprocessing.StandardScaler().fit_transform(featuresData_)
        #featureData = preprocessing.MinMaxScaler().fit_transform(featuresData_)
        labelsData = labelsData_ / labelDiv
        return featureData, labelsData
        
    #[2020-02, 2021-10)
    def __getData(self):
        self.__checkLoadedData()
        beginDays = self.__countDaysFromMonthPeriod(self.featureDataMonthStartStr, self.dataMonthStartStr)
        days = self.__countDaysFromMonthPeriod(self.dataMonthStartStr, self.dataMonthEndStr)
        print("total days: %d\t, begin: %d\t, end: %d\t" % (self.__countDaysFromMonthPeriod(self.featureDataMonthStartStr, self.featureDataMonthEndStr), beginDays, beginDays+days))
        feturesData, labelsData = self.featuresData[beginDays:beginDays+days], self.labelsData[beginDays:beginDays+days]
        return feturesData, labelsData
    
    def getData(self):
        featuresData, labelsData = self.__getData()
        return self.__dataPreprocess(featuresData, labelsData, labelDiv=1e9)
    
    def getWeeklyData(self):
        featuresData, labelsData = self.__getData()
        print("shape for labelsData: %s" % str(labelsData.shape))
        days, features = featuresData.shape
        weeks = days//7
        feturesDataWeekly = np.empty((weeks, features))
        labelsDataWeekly = np.empty((weeks))
        #print("WeeklyData processing...")
        for i in range(weeks):
            dayIndexEnd = days - i*7
            weekIndex = weeks - i - 1
            feturesDataWeekly[weekIndex] = featuresData[dayIndexEnd-7:dayIndexEnd].sum(axis=0)
            labelsDataWeekly[weekIndex] = labelsData[dayIndexEnd-7:dayIndexEnd].sum()
            #print("One record: %s %s" % (str(featuresData[dayIndexEnd-7:dayIndexEnd].sum(axis=0)), str(labelsData[dayIndexEnd-7:dayIndexEnd].sum())))
        return self.__dataPreprocess(feturesDataWeekly, labelsDataWeekly, labelDiv=5e9)
    
    #Incooperate more features into self.featuresData, for future work
    #You can call:
    #   dataManager = DataManager(dataFrom="./data")
    #   dataManager.expendFeatures(featureList)
    #   data = dataManager.getData()
    #   #in this way, the extra feature will be fully used by model training without DimensionalityReduction
    #   data = DataDimensionalityReduction().reduceDimension(data[:,:(len(data)-len(featureList))])
    #featureList: [ndarray_1D, ndarray_1D_2, ndarray_1D_3, ...]
    '''
    def expandFeatures(self, featureList, startMonth="2020-02", endMonth="2021-10"):
        oldFeatures = None
        if len(featureList) == len(self.featuresData):
            oldFeatures = self.featuresData
        else:
            startDayIndex = self.__countDaysFromMonthPeriod(self.dataMonthStartStr, startMonth)
            days = self.__countDaysFromMonthPeriod(startMonth, endMonth)
            for i in range(len(featureList)):
                assert days == len(featureList[i]), "Illegal parameters"
            oldFeatures = self.featuresData[startDayIndex:startDayIndex + days]
            #oldFeatures = self.featuresData[startDayIndex:startDayIndex + days,:-1]
            #if self.featuresData.shape[1] == 2:
            #    oldFeatures = oldFeatures[:,None]
        self.featuresData = np.concatenate(([oldFeatures] + [featureList[i][:,None] for i in range(len(featureList))]), axis=-1)
        self.labelsData = self.labelsData[startDayIndex:startDayIndex+days]
        self.__checkLoadedData()
        return self.featuresData, self.labelsData
    '''
    
    #featureList should contain data from 2018-09 to 2021-09
    def expandFeatures(self, featureList):
        if len(featureList) != 0:
            for i in range(len(featureList)):
                assert len(featureList[i]) == len(self.featuresData), "illegal paramters"
            # /7.0 for weekly consumer confidence addition
            self.featuresData = np.concatenate(([self.featuresData] + [(featureList[i][:,None].astype(float)) / 7.0 for i in range(len(featureList))]), axis=-1)
            self.__checkLoadedData()
        return self
'''
#for autoencoder    
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
        
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
        shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)
        
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dense": self.dense,
            "activation": self.activation,
        })
        return config
'''
class DataDimensionalityReduction:
    # pass a trained encoder (path) or nothing
    def __init__(self, featureNum=None, encoder=None, encoderPath='./', isEncoderTraining = False):
        self.encoder = encoder
        self.encoderPath = encoderPath
        self.isEncoderTraining = isEncoderTraining
        if not isEncoderTraining:
            if encoder == None:
                self.encoder = keras.models.load_model(os.path.join(encoderPath, "autoencoder.h5"))
        else:
            #build an default autoencoder model
            assert featureNum != None, "Illegal parameters"
            encoder = keras.models.Sequential([
                keras.layers.Dense(featureNum, activation="selu", input_shape=[featureNum]),
                keras.layers.Dense(featureNum//2, activation="selu"),
                keras.layers.Dense(featureNum//4, activation="selu")
                #keras.layers.ActivityRegularization(l1=1e-3)
            ])
            decoder = keras.models.Sequential([
                keras.layers.Dense(featureNum//2, activation="selu", input_shape=[featureNum//4]),
                keras.layers.Dense(featureNum, activation="selu", input_shape=[featureNum//2]),
                keras.layers.Dense(featureNum, activation="sigmoid")
            ])
            autoEncoder = keras.models.Sequential([encoder, decoder])
            autoEncoder.compile(loss="mse", optimizer=keras.optimizers.Adam())
            earlyStoppingCB = callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
            #checkpointCB = callbacks.ModelCheckpoint("autoencoder.h5", monitor="val_binary_crossentropy", save_best_only=True)
            self.callbacks = [earlyStoppingCB]
            self.model = autoEncoder
            self.encoder = encoder
    # 2D featureData: (samples, features)
    def reduceDimension(self, featureData):
        if self.isEncoderTraining:
            size = len(featureData)
            indexes = np.array([i for i in range(size)])
            random.shuffle(indexes)
            split = int(size*0.8)
            X_train = featureData[indexes[:split]]
            X_valid = featureData[indexes[split:]]
            self.model.fit(X_train, X_train, epochs=2000, validation_data=(X_valid, X_valid), callbacks=self.callbacks)
            #save encoder, not model
            self.encoder.save(os.path.join(self.encoderPath, "autoencoder.h5"))
        return self.encoder.predict(featureData)

# make the 2D data(ndarray, X) transferred to 3D time-series data(samples, timeStepPerSample, features) for X
# 1. Dimension Transfer, 2. preprocess labelsArray(cashing_sum) (value: 0 ~ 1.0)
class DimensionTransfer:
    #default: use 180 days(time steps) as a sample, for each time_step, predict the cashing_sum for next 60 days
    def __init__(self, timeStepPerSample, nextTimeStepPerPrediction):
        self.timeStepPerSample = timeStepPerSample
        self.nextTimeStepPerPrediction = nextTimeStepPerPrediction
    
    def __checkData(self, X, y, XForLastPrediction):
        print("shape for X: %s\nshape for y: %s\nshape for XForLastPrediction: %s\n" % (X.shape, y.shape, XForLastPrediction.shape))
        print("Data transferred\n")
        print(X)
        print(y)
    
    #use transferData(featuresArray[-(testSize+nextTimeStepPerPrediction):], labelsArray[-(testSize+nextTimeStepPerPrediction):]) to generate X_test and y_test
    #Or use transferData(featuresArray, labelsArray) to generate X and y without TimeGAN application
    def transferData(self, featuresArray, labelsArray, testSize, validateRate):
        totalTimeStep, featurePerTimeStep = featuresArray.shape
        sampleSize = totalTimeStep-self.timeStepPerSample-self.nextTimeStepPerPrediction+1
        newX = np.empty((sampleSize, self.timeStepPerSample, featurePerTimeStep+1))
        newY = np.empty((sampleSize, self.timeStepPerSample, self.nextTimeStepPerPrediction))
        for i in range(sampleSize):
            for j in range(self.timeStepPerSample):
                newX[i,j,:-1] = featuresArray[i+j,:]
                newX[i,j,-1] = labelsArray[i+j]        
                start = i+j+1
                newY[i,j,:] = labelsArray[start:start+self.nextTimeStepPerPrediction]
        # XForLastPrediction has no labels, the prediction from XForLastPrediction is for the future
        # shape for XForLastPrediction should be (1, nextTimeStepPerPrediction, features+1) or (1, timeStepPerSample, features+1)
        #XForLastPrediction = np.concatenate([featuresArray[-self.nextTimeStepPerPrediction:], labelsArray[-self.nextTimeStepPerPrediction:][:,None]], axis=-1)[None,:]    
        XForLastPrediction_ = np.concatenate([featuresArray[-self.timeStepPerSample:], labelsArray[-self.timeStepPerSample:][:,None]], axis=-1)[None,:]    
        print(featuresArray.dtype)
        print(labelsArray.dtype)
        print("XForLastPrediction data type")
        print(XForLastPrediction_.dtype)
        XForLastPrediction = XForLastPrediction_.astype(newX.dtype)
        #print(XForLastPrediction.dtype)
        self.__checkData(newX, newY, XForLastPrediction)
        X_train, y_train, X_valid, y_valid  = self.splitData(newX[:-testSize], newY[:-testSize], validateRate)
        return X_train, y_train, X_valid, y_valid, newX[-testSize:], newY[-testSize:], XForLastPrediction
    
    #call transferToTimeSeries first to get timeSeriesData data for TimeGAN training, then convertTimeSeriesData to get X and y for model training
    #methods for TimeGAN training and synthetic time series generation
    #testing data is excluded
    def transferToTimeSeries(self, featuresArray, labelsArray, testSize):
        sampleTimeSteps = self.timeStepPerSample + self.nextTimeStepPerPrediction
        totalTimeStep, featurePerTimeStep = featuresArray.shape
        # minus nextTimeStepPerPrediction to exclude last 3 month data in training set for testing
        sampleSize = totalTimeStep - sampleTimeSteps - self.nextTimeStepPerPrediction - testSize + 1
        assert sampleSize > 0, "illegal parameters"
        timeSeriesData = np.empty((sampleSize, sampleTimeSteps, featurePerTimeStep+1))
        for i in range(sampleSize):
            for j in range(sampleTimeSteps):
                timeSeriesData[i,j,:-1] = featuresArray[i+j]
                timeSeriesData[i,j,-1] = labelsArray[i+j]
        print("TimeSeries Data transferred")
        print(timeSeriesData.shape)
        return timeSeriesData
    
    #convert (samples, XtimeSteps + YtimeSteps, features) to X: (samples, XtimeSteps, features) and y: (samples, XtimeSteps, predictionForEachYtimeSteps)
    #X, y used only for training
    def convertTimeSeriesData(self, timeSeriesData, syntheticDataList):
        X_all = np.concatenate(syntheticDataList + [timeSeriesData], axis=0)
        sampleSize, totalTimeStep, featurePerTimeStep = X_all.shape
        assert totalTimeStep == self.timeStepPerSample + self.nextTimeStepPerPrediction, "illegal parameters"
        X = np.empty((sampleSize, self.timeStepPerSample, featurePerTimeStep))
        y = np.empty((sampleSize, self.timeStepPerSample, self.nextTimeStepPerPrediction))
        for i in range(sampleSize):
            X[i] = X_all[i, :self.timeStepPerSample]
            for j in range(self.timeStepPerSample):
                y[i,j] = X_all[i,j+1:j+1+self.nextTimeStepPerPrediction,-1]
        return X, y
    
    def splitData(self, X, y, valDataRate):
        assert len(X) == len(y), "illegal parameters"
        valDataSize = int(valDataRate * len(X))
        indexes = np.random.permutation(len(X))
        X_valid = X[indexes[:valDataSize]]
        y_valid = y[indexes[:valDataSize]]
        X_train = X[indexes[valDataSize:]]
        y_train = y[indexes[valDataSize:]]
        print("Data splited")
        print("X_train %s\ty_train: %s\nX_valid: %s\ty_valid: %s\n" % (X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))
        return X_train, y_train, X_valid, y_valid 
    '''
    #saveData(dataTo="./", dataList=dataList, nameList=["X_test.npy", "y_test.npy", "XForLastPrediction.npy", "timeSeriesData.npy"])
    def saveData(self, dataTo, dataList, nameList):
        assert len(dataList) == len(nameList), "illegal parameters"
        for i in range(len(dataList)):
            np.save(os.path.join(dataTo, nameList[i]), dataList[i])
    # X_test, y_test, XForLastPrediction, timeSeriesData = loadData(dataFrom="./", nameList=["X_test.npy", "y_test.npy", "XForLastPrediction.npy", "timeSeriesData.npy"])
    def loadData(self, dataFrom, nameList):
        dataList = []
        for i in range(len(dataList)):
            dataList.append(np.load(os.path.join(dataFrom, nameList[i])))
        return dataList
    '''
    
# A naive model    
class CNNLSTM:
    def __init__(self, featurePerTimeStep=None, outputNum=None, epochs=50, batchSize=32, extraCallbackList=[], isLoadModel=False, modelFrom=None):
        self.batchSize = batchSize
        self.earlyStoppingCB = callbacks.EarlyStopping(monitor="val_lastTimeStepMSEError", patience=30, restore_best_weights=True)
        self.checkpointCB = callbacks.ModelCheckpoint("CNNLSTM_only_covid_extra_feature.h5", monitor="val_lastTimeStepMSEError", save_best_only=True)
        self.callbacks = [self.earlyStoppingCB, self.checkpointCB] + extraCallbackList
        self.epochs = epochs
        if isLoadModel:
            self.model = self.__loadModel(os.path.join(modelFrom, "CNNLSTM_only_covid_extra_feature.h5"))
        else:
            assert (featurePerTimeStep!=None and outputNum!=None), "illegal parameters"
            self.model = None
            self.model = self.__buildModel(featurePerTimeStep, outputNum)
    
    def lastTimeStepMAEError(self, y_true, y_pred):
        return keras.metrics.mae(y_true[:, -1], y_pred[:,-1])
    
    def lastTimeStepMSEError(self, y_true, y_pred):
        return keras.metrics.mse(y_true[:, -1], y_pred[:,-1]) 
    
    def lastTimeStepMAPEError(self, y_true, y_pred):
        return keras.metrics.mape(y_true[:, -1], y_pred[:,-1]) 
    
    def __buildModel(self, featurePerTimeStep, outputNum):
        input_ = layers.Input(shape=[None, featurePerTimeStep])
        prevLayer = input_
        #for dilation in (1, 2, 4, 7, 14, 28):
        for dilation in (1, 2, 4, 6, 8, 12, 16):
            prevLayer = layers.Conv1D(filters=60, kernel_size=2, padding="causal", 
                                      activation="relu", 
                                      dilation_rate=dilation)(prevLayer)
        lstm1 = layers.LSTM(120, return_sequences=True)(prevLayer)
        lstm2 = layers.LSTM(120, return_sequences=True)(lstm1)
        lstm3 = layers.LSTM(60, return_sequences=True)(lstm2)
        outputLayer = layers.TimeDistributed(layers.Dense(outputNum))(lstm3)
        model = keras.Model(inputs=[input_], outputs=[outputLayer])
        #model.compile(loss=keras.losses.Huber(delta=2.5), optimizer=keras.optimizers.Adam(learning_rate=0.0005), metrics=[self.lastTimeStepMAEError, self.lastTimeStepMSEError, self.lastTimeStepMAPEError])
        model.compile(loss="mse", optimizer="adam", metrics=[self.lastTimeStepMAEError, self.lastTimeStepMSEError, self.lastTimeStepMAPEError])
        #model.compile(loss=keras.losses.Huber(delta=2.5), optimizer="adam", metrics=[self.lastTimeStepError])
        return model
    
    def __loadModel(self, modelPath):
        #model = keras.models.load_model(modelPath, custom_objects={"lastTimeStepError": self.lastTimeStepError}, compile=False)
        model = keras.models.load_model(modelPath, compile=False)
        model.compile(loss="mse", optimizer="adam", metrics=[self.lastTimeStepMAEError, self.lastTimeStepMSEError, self.lastTimeStepMAPEError])
        return model
    
    def getModel(self):
        return self.model
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        
        return self.model.fit(X_train, y_train, epochs=self.epochs, steps_per_epoch=X_train.shape[0]//self.batchSize+1, validation_data=(X_valid, y_valid), callbacks=self.callbacks)

    # Once the training is stopped by accident, it can be recovered from file "CNNLSTM.h5"
    # You can call: 
    #    cnnLSTM = CNNLSTM(isLoadModel=True, modelFrom="./model/")
    #    cnnLSTM.resumeTraining( X_train, y_train, X_valid, y_valid, leftEpochs) 
    # To recover the left model training
    def resumeTraining(self, X_train, y_train, X_valid, y_valid):
        return self.fit(X_train, y_train, X_valid, y_valid)
        

# For tuning hyperparameters, not finished yet
class WalkForwardValidation:
    def __init__(self):
        pass

class WeekToDaysDecoder:
    # Train a new WeekToDaysDecoder
    #weekToDaysDecoder = WeekToDaysDecoder()
    #weekToDaysDecoder.train(weeklyX, weeklyY, weeks, dayEnd="2021-07-02")
    # or just loading: 
    #weekToDaysDecoder = WeekToDaysDecoder(isLoadModel=True)
    #prediction = weekToDaysDecoder.predict(self, weeklyX, weeks, dayEnd="2021-10-01"):
    def __init__(self, isLoadModel=False, modelFrom='./', modelTo='./'):
        self.featureNum = 1+12+6+4*7
        self.earlyStoppingCB = callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
        #From: https://www.timeanddate.com/holidays/japan/
        self.holidays = {"2018-09-17", "2018-09-23", "2018-09-24", "2018-10-08", "2018-11-03",
                         "2018-11-15", "2018-11-23", "2018-12-22", "2018-12-23", "2018-12-24",
                         "2018-12-25", "2018-12-31", "2019-01-01", "2019-01-02", "2019-01-03",
                         "2019-01-14", "2019-02-11", "2019-02-14", "2019-03-03", "2019-03-21",
                         "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03",
                         "2019-05-04", "2019-05-05", "2019-05-06", "2019-06-22", "2019-07-07",
                         "2019-07-15", "2019-08-06", "2019-08-09", "2019-08-11", "2019-08-12",
                         "2019-09-16", "2019-09-23", "2019-10-14", "2019-10-22", "2019-11-03",
                         "2019-11-04", "2019-11-15", "2019-11-23", "2019-12-22", "2019-12-25",
                         "2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-13",
                         "2020-02-11", "2020-02-14", "2020-02-23", "2020-02-24", "2020-03-03",
                         "2020-03-20", "2020-04-29", "2020-05-03", "2020-05-04", "2020-05-05", 
                         "2020-05-06", "2020-06-21", "2020-07-07", "2020-07-23", "2020-07-24", 
                         "2020-08-06", "2020-08-09", "2020-08-10", "2020-09-21", "2020-09-22", 
                         "2020-11-03", "2020-11-15", "2020-11-23", "2020-12-21", "2020-12-25",
                         "2020-12-31", "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-11",
                         "2021-01-14", "2021-02-23", "2021-03-03", "2021-03-20", "2021-04-29", 
                         "2021-05-03", "2021-05-04", "2021-05-05", "2021-06-21", "2021-07-07", 
                         "2021-07-22", "2021-07-23", "2021-08-06", "2021-08-08", "2021-08-09",
                         "2021-09-20", "2021-09-23"}
        if not isLoadModel:
            decoder = keras.models.Sequential([
                keras.layers.Dense(self.featureNum//2, activation="selu", input_shape=[self.featureNum]),
                keras.layers.Dense(self.featureNum, activation="selu"),
                keras.layers.Dense(self.featureNum//2, activation="selu"),
                keras.layers.Dense(7)
            ])
            decoder.compile(loss="mse", optimizer=keras.optimizers.Adam())
            self.model = decoder
        else:
            self.model = keras.models.load_model(os.path.join(modelFrom, "weekToDaysDecoder.h5"))
        
    def __one_hot(self, total, index):
        res = np.zeros((total,)) + 0.01
        if index >= 0:
            res[index] = 0.99
        return res
    
    def __isHoliday(self, day):
        dayStr = day.strftime("%Y-%m-%d")
        return dayStr in self.holidays
    
    def __isConsecutiveHoliday(self, day):
        #day.weekday() start from 0
        weekday = day.weekday()+1
        if weekday == 5 or weekday == 1 or self.__isHoliday(day+relativedelta(days=1)) or self.__isHoliday(day-relativedelta(days=1)):
            return True
        return False
    #return value: 0~5 
    def __getWeekOfMonth(self, day):
        year, month, d = day.year, day.month, 1
        date = datetime.strptime("%d-%02d-%02d" % (year,month,d), "%Y-%m-%d")
        extraDays = 7 - date.weekday()
        baseWeek = 1
        if extraDays == 7:
            extraDays = 0
            baseWeek = 0
        days = day.day
        if days <= extraDays:
            return 0
        assert (days-extraDays-day.weekday()-1) % 7 == 0
        return (days-extraDays-1) // 7 + baseWeek
    
    #dayEnd is excluded [startDay, endDay)
    def __genTimeFeautures(self, weeklyX, weeks, dayEndStr):
        featuresArray = np.empty((weeks, self.featureNum))
        endDay = datetime.strptime(dayEndStr, "%Y-%m-%d")
        #the first day for this week
        crtDay = endDay - relativedelta(days=7)
        crtWeek = weeks-1
        while crtWeek >= 0:
            featuresArray[crtWeek][0] = weeklyX[crtWeek]
            #month
            featuresArray[crtWeek][1:13] = self.__one_hot(12, crtDay.month-1)
            #week of month
            featuresArray[crtWeek][13:19] = self.__one_hot(6, self.__getWeekOfMonth(crtDay))
            #holiday
            tmpDay = crtDay
            for i in range(1,8):
                if (tmpDay.month == 12 and tmpDay.day <= 31 and tmpDay.day >=24) or (tmpDay.month == 1 and tmpDay.day <= 3 and tmpDay.day >=1):
                    holiday_type = 0
                elif (tmpDay.month == 4 and tmpDay.day <= 31 and tmpDay.day >=29) or (tmpDay.month == 5 and tmpDay.day <= 5 and tmpDay.day >=1):
                    holiday_type = 1
                elif self.__isHoliday(tmpDay):
                    #holidays for 3 consecutive days including weekends
                    if self.__isConsecutiveHoliday(tmpDay):
                        holiday_type = 2
                    else:
                        holiday_type = 3
                else:
                    holiday_type = -1
                featuresArray[crtWeek][19+(i-1)*4:19+i*4] = self.__one_hot(4, holiday_type)
                tmpDay += relativedelta(days=1)
            crtDay -= relativedelta(days=7)
            crtWeek -= 1
        return featuresArray;
    
    def __splitData(self, X, y, validRatio=0.2):
        assert len(X) == len(y), "illegal parameters"
        indexes = np.random.permutation(len(X))
        splitIndex = int(validRatio*len(X))
        X_train = X[indexes[:-splitIndex]]
        y_train = y[indexes[:-splitIndex]]
        X_valid = X[indexes[-splitIndex:]]
        y_valid = y[indexes[-splitIndex:]]
        return X_train, X_valid, y_train, y_valid
    
    #assume the factors effect each day equally.
    def train(self, weeklyX, weeklyY, weeks, dayEnd="2021-07-02"):
        X = self.__genTimeFeautures(weeklyX, weeks, dayEnd)
        print("genetated X in WeekToDaysDecoder")
        print(X)
        X_train, X_valid, y_train, y_valid = self.__splitData(X, weeklyY, 0.2)
        self.model.fit(X_train, y_train, epochs=5000, validation_data=(X_valid, y_valid), callbacks=[self.earlyStoppingCB])
        
    def predict(self, weeklyX, weeks, dayEnd="2021-10-01"):
        X = self.__genTimeFeautures(weeklyX, weeks, dayEnd)
        return self.model.predict(X)   

#------------------------------------------extra features------------------------------------------#
#extract data during 2020-02 ~ 2021-11, 22 months, based on per month
#however the training data is during 2020-02 ~ 2021-09, 20 months, based on per day
class ConsumerConfidenceExtractor:
    def __init__(self, path, prev_month_start=39, prev_month_end=2 ,extractedColumnIndexes=[]):
        self.month_data = []
        df = pd.read_csv(path)
        df = df.iloc[-prev_month_start:-prev_month_end]
        if len(extractedColumnIndexes) > 0:
            for index in extractedColumnIndexes:
                self.month_data.append(df.loc[:, df.columns[index]].to_numpy())
        else:
            array = df.to_numpy()
            self.month_data = [array[:,i] for i in array.shape[1]]
            
    # return a list with each element being an ndarray as a feature column       
    def getData(self):
        return self.month_data

class MonthlyFeatureToDailyFeature:
    #[monthStartStr, monthEndStr]
    def __init__(self, monthStartStr, monthEndStr, monthFeatureList):
        self.monthStart = datetime.strptime(monthStartStr, "%Y-%m")
        self.monthEnd = datetime.strptime(monthEndStr, "%Y-%m") + relativedelta(months=1)
        self.monthFeatureList = monthFeatureList
        
    def convertData(self):
        featureList = [[] for i in range(len(self.monthFeatureList))]
        monthCrt = self.monthStart
        monthIndex = 0
        nextMonth = monthCrt + relativedelta(months=1)
        while monthCrt != self.monthEnd:
            days = (nextMonth - monthCrt).days
            for featureIndex in range(len(self.monthFeatureList)):
                for day in range(days):
                    featureList[featureIndex].append(self.monthFeatureList[featureIndex][monthIndex])
            monthIndex += 1
            monthCrt = nextMonth
            nextMonth = monthCrt + relativedelta(months=1)
        assert (self.monthEnd - self.monthStart).days == len(featureList[0]), "Illegal parameters"
        return [np.array(featureList[i]) for i in range(len(self.monthFeatureList))]
'''    
class DateFeatureGenerator:
    def __init__(self, startDateStr, endDateStr, formatStr="%Y-%m-%d"):
        self.startDate = datetime.strptime(startDateStr, formatStr)
        self.endDate = datetime.strptime(endDateStr, formatStr) + relativedelta(days=1)
        self.format = formatStr
    
    def __week_of_month(self, date_value):
        value = (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)
        if value < 0:
            value += 52
        return value
    
    def genDateFeatureList(self):
        dates = []
        crtDate = self.startDate
        while crtDate != self.endDate:
            dates.append(crtDate)
            crtDate += relativedelta(days=1)
        df = pd.DataFrame(np.array(dates)[:,None])
        print("df")
        print(df)
        df['date'] = df[0] # index: DatetimeIndex
        df['day_of_week'] = df['date'].dt.dayofweek.values
        df['quarter'] = df['date'].dt.quarter.values
        df['month'] = df['date'].dt.month.values
        #df['year'] = df['date'].dt.year.values
        df['day_of_year'] = df['date'].dt.dayofyear.values
        df['day_of_month'] = df['date'].dt.day.values
        df['week_of_year'] = df['date'].dt.weekofyear.values
        df['week_of_month'] = df.apply(lambda x: self.__week_of_month(x.date), axis = 1)
        #X = df[['day_of_week', 'quarter', 'month', 'year', 'day_of_year', 'day_of_month', 'week_of_year','week_of_month']]
        X = df[['day_of_week', 'quarter', 'month', 'day_of_year', 'day_of_month', 'week_of_year','week_of_month']]
        print("X")
        print(X)
        featureList = []
        for columnStr in X.columns:
             featureList.append(df[columnStr].to_numpy())
        print("Date Feature List")
        print(featureList)
        return featureList    
'''  

class Main:
    def __init__(self, dataFrom, extraFeatures=[], labelFrom=None, isCreatefeatureDict=False, isCreateData=False, isResumeTraining=False, isTimeGANTraining=False, isAutoEncoderTraining=False, isTimeGANEnabled=False, isLoadEncoder=False, leftEpochs=50, modelFrom=None, dataTo=None):
        self.leftEpochs = leftEpochs
        X_test, y_test, timeSeriesData, X, y, XForLastPrediction, model = [None]*7
        #weekly
        timeStepPerSampleWeekly = 39
        nextTimeStepPerPredictionWeekly = 13
        testSizeWeekly = 13
        validationRate = 0.2
        # assume timeStepPerSample >= nextTimeStepPerPrediction
        timeStepPerSampleDaily = timeStepPerSampleWeekly*7
        nextTimeStepPerPredictionDaily = nextTimeStepPerPredictionWeekly*7
        testSizeDaily = nextTimeStepPerPredictionWeekly*7
        
        timeStepPerSample = timeStepPerSampleWeekly
        nextTimeStepPerPrediction = nextTimeStepPerPredictionWeekly
        testSize = testSizeWeekly
        
        dimensionTransferDaily = DimensionTransfer(timeStepPerSample, nextTimeStepPerPrediction)
        dimensionTransferWeekly = DimensionTransfer(timeStepPerSampleWeekly, nextTimeStepPerPredictionWeekly)
        dimensionTransfer = dimensionTransferWeekly

        featureDict, invalidFeatureDict = None, None
        if isCreatefeatureDict:
            featureDict, featureNum, invalidFeatureDict = FeatureLabelExplorer(dataFrom, dataTo).exploreLabel()
        else:
            featureDict, featureNum,invalidFeatureDict = FeatureLabelExplorer(labelFrom).loadfeatureDict()
        if isCreateData:
            dataManager = DataManager(dataFrom, dataMonthStart="2018-09", dataMonthEnd="2021-10", isConvertData=True, featureDict=(featureDict, featureNum, invalidFeatureDict), dataTo=dataTo)
        else:
            dataManager = DataManager(dataFrom, dataMonthStart="2018-09", dataMonthEnd="2021-10", isConvertData=False)
        #add more features
        dataManager.expandFeatures(extraFeatures)
        #featuresArrayDaily, labelsArrayDaily = dataManager.getData()
        featuresArrayWeekly, labelsArrayWeekly = dataManager.getWeeklyData()

        #training or loading autoencoder
        #newFeaturesArray = DataDimensionalityReduction(featureNum=featuresArray.shape[1], isEncoderTraining=isAutoEncoderTraining).reduceDimension(featuresArray)

        #weekly feature array
        newFeaturesArray = featuresArrayWeekly
        labelsArray = labelsArrayWeekly
        #newFeatureSArray = DataDimensionalityReduction(featureNum=featuresArrayWeekly.shape[1], isEncoderTraining=isAutoEncoderTraining).reduceDimension(featuresArrayWeekly)
        print("converted FeaturesArray shape:")
        print(newFeaturesArray.shape)
        print(newFeaturesArray)
        if isTimeGANEnabled:
            #timeSeriesData excludes testing data 
            #generate only time series data after Covid-19(2020-01)
            featuresArrayForTimeGAN, labelsArrayForTimeGAN = DataManager(dataFrom, dataMonthStart="2018-09", dataMonthEnd="2021-10", isConvertData=False).expandFeatures(extraFeatures).getWeeklyData()
            timeSeriesData = dimensionTransfer.transferToTimeSeries(featuresArrayForTimeGAN, labelsArrayForTimeGAN, testSize)
            X_train_real, y_train_real, X_valid_real, y_valid_real, X_test, y_test, XForLastPrediction = dimensionTransfer.transferData(newFeaturesArray, labelsArray, testSize, validationRate)
            #parameters for TimeGAN
            seq_len = timeStepPerSample + nextTimeStepPerPrediction
            n_seq = timeSeriesData.shape[2]
            hidden_dim=120
            gamma=1
            noise_dim = int(n_seq*4)
            layers_dim = 128
            batch_size = 14
            learning_rate = 5e-4
            gan_args = ModelParameters(batch_size=batch_size, lr=learning_rate, noise_dim=noise_dim, layers_dim=layers_dim)
            # Loading or training TimeGAN
            if not isTimeGANTraining:
                timeGAN = TimeGAN.load(os.path.join(dataFrom, 'TimeGAN_100epoch_consumer_confidence.pkl'))
            else:
                timeGAN = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=gamma)
                #shape for "stock_data": (samples, time_steps, features), time series data
                timeGAN.train(timeSeriesData, train_steps=1000)
                timeGAN.save(os.path.join(dataTo, 'TimeGAN.pkl'))
            extraDataCount = 4
            X_List = [timeGAN.sample(len(timeSeriesData)) for i in range(extraDataCount)] + [timeSeriesData]
            print("synthetic data shape %s" % str(X_List[0].shape))
            print("real data shape %s" % str(timeSeriesData.shape))
            X, y = dimensionTransfer.convertTimeSeriesData(timeSeriesData, X_List)
            X_train_gen, y_train_gen, X_valid_gen, y_valid_gen = dimensionTransfer.splitData(X, y, validationRate)
            X_train = np.concatenate([X_train_real, X_train_gen], axis=0)
            X_valid = np.concatenate([X_valid_real, X_valid_gen], axis=0)
            y_train = np.concatenate([y_train_real, y_train_gen], axis=0)
            y_valid = np.concatenate([y_valid_real, y_valid_gen], axis=0)
            indexes_train = np.random.permutation(len(X_train))
            indexes_valid = np.random.permutation(len(X_valid))
            X_train = X_train[indexes_train]
            X_valid = X_valid[indexes_valid]
            y_train = y_train[indexes_train]
            y_valid = y_valid[indexes_valid]
            #dimensionTransfer.saveData(dataTo=dataTo, dataList=[X_train, y_train, X_valid, y_valid, X_test, y_test, XForLastPrediction, timeSeriesData], nameList=["X_train_t.npy", "y_train_t.npy", "X_valid_t.npy", "y_valid_t.npy", "X_test_t.npy", "y_test_t.npy", "XForLastPrediction_t.npy", "timeSeriesData.npy"])
        else:
            X_train, y_train, X_valid, y_valid, X_test, y_test, XForLastPrediction = dimensionTransfer.transferData(newFeaturesArray, labelsArray, testSize, validationRate)
            #dimensionTransfer.saveData(dataTo=dataTo, dataList=[X_train, y_train, X_valid, y_valid, X_test, y_test, XForLastPrediction], nameList=["X_train.npy", "y_train.npy", "X_valid", "y_valid", "X_test.npy", "y_test.npy", "XForLastPrediction.npy"])

        model,history = None,None
        if isResumeTraining:
            model = CNNLSTM(X_train.shape[2], nextTimeStepPerPrediction, isLoadModel=True, modelFrom=modelFrom, epochs=leftEpochs)
            history = model.resumeTraining(X_train, y_train, X_valid, y_valid)
        else:
            model = CNNLSTM(X_train.shape[2], nextTimeStepPerPrediction, epochs=leftEpochs)
            history = model.fit(X_train, y_train, X_valid, y_valid)
        model = model.getModel()
        print(model.summary())
        model.evaluate(X_test, y_test)
        #meanLoss, lastLoss = model.evaluate(X_test, y_test)
        losses = model.evaluate(X_test, y_test)
        print("losses")
        print(losses)
        errorDict = {}
        keys = ["MAE", "MSE", "MAPE"]
        for i in range(len(keys)):
            errorDict[keys[i]] = losses[i+1]
        print("lastTimeStepError:\nMAE: %.12f\tMSE: %.12f\tMAPE: %.12f" % (losses[1], losses[2], losses[3]))
        baseMAE, baseMSE, baseMAPE = self.__genBaselineError(X_test, y_test, nextTimeStepPerPrediction)
        print("BaselineError:\nMAE: %.12f\tMSE: %.12f\tMAPE: %.12f" % (baseMAE, baseMSE, baseMAPE))

        #make prediction for the following 3 months
        predictionForFuture = model.predict(XForLastPrediction)
        '''
        print("\npredictionForFuture")
        print(predictionForFuture)
        '''
        predictionForYTest = model.predict(X_test)
        '''
        print("\npredictionForYTest")
        print(predictionForYTest)
        print(predictionForYTest[-1,-1,:])
        print("\ny_test")
        print(y_test)
        print(y_test[-1,-1,:])
        '''
        dataManager = DataManager(dataFrom, dataMonthStart="2018-09", dataMonthEnd="2021-10", isConvertData=False).expandFeatures(extraFeatures)
        features_weekly, cashing_sum_weekly = dataManager.getWeeklyData()
        features_daily, cashing_sum_daily = dataManager.getData()
        weeklyX = cashing_sum_weekly
        weeks = len(cashing_sum_weekly)
        weeklyY = np.empty((weeks, 7))
        for i in range(1, weeks+1):
            weeklyY[-i] = cashing_sum_daily[len(cashing_sum_daily)-7*i:len(cashing_sum_daily)-7*(i-1)]
        weekToDaysDecoder = WeekToDaysDecoder()
        if not isLoadEncoder:
            weekToDaysDecoder.train(weeklyX, weeklyY, weeks=weeks, dayEnd="2021-07-02")
        dailyPrediction_ = weekToDaysDecoder.predict(predictionForYTest[-1,-1,:], weeks=len(predictionForYTest[-1,-1,:]), dayEnd="2021-10-01")
        dailyPrediction = dailyPrediction_.flatten()
        dailyReal = cashing_sum_daily[-91:]
        self.__plotResults(history, predictionForYTest[-1,-1,:], y_test[-1,-1,:], predictionForFuture[-1,-1,:], errorDict) 
        #plot daily prediction
        errorDict2 = {}
        errorDict2["MAE"] = keras.metrics.mae(dailyPrediction, dailyReal)
        errorDict2["MSE"] = keras.metrics.mse(dailyPrediction, dailyReal)
        errorDict2["MAPE"] = keras.metrics.mape(dailyPrediction, dailyReal)
        self.__plotSingleImage([i for i in range(1, 92)], [dailyPrediction, dailyReal], labels=["Prediction", "Real value"], title="Prediction Comparasion for Last 91days", figSize=(18,6), errorDict=errorDict2, yLabel = 'daily Cashing sum(1e9)')
    
    def __genBaselineError(self, X_test, y_test, nextTimeStepPerPrediction):
        # The cashing_sum for last 60(nextTimeStepPerPrediction) time_steps is used as the prediction for next 60 time_step
        y_pred = X_test[:, -nextTimeStepPerPrediction:,-1].flatten()
        error = [keras.metrics.mae(y_pred, y_test[:,-1].flatten()), keras.metrics.mse(y_pred, y_test[:,-1].flatten()), keras.metrics.mape(y_pred, y_test[:,-1].flatten())]
        return error
    
    def __plotSingleImage(self, X, yList, labels, errorDict={}, title='', figSize=(12,6), yLabel=''):
        plt.figure(figsize=figSize)
        ax = plt.gca()
        ax.set_title(title)
        ax.set_xlabel('Day')
        ax.set_ylabel(yLabel)
        #ax.set_ylim(ymin=0.0, ymax=1.0)
        for i in range(len(yList)):
            plt.plot(X, yList[i], label=labels[i])
        text = ''.join(["%s: %.12f\n" % (key, value) for key, value in errorDict.items()])
        ax.text(0.1, 0.1, text)
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    #y_pred_test: ndarray_1D, last_time_step_predictions
    #errorDict: mae:..., mse:..., mape:...
    def __plotResults(self, history, y_pred_test, y_test, y_pred_future, errorDict):
        if self.leftEpochs == 0:
            self.__history_half_shift_plot(history)
        else:
            self.__history_half_shift_plot(history, shifted_column_names=["val_loss", "val_lastTimeStepMAEError"])
        X = np.linspace(1, len(y_test), len(y_test))
        self.__plotSingleImage(X, [y_pred_test, y_test], errorDict=errorDict, title='Prediction Comparasion for Last 13 weeks(91days, testing data)', labels=["Prediction", "Real value"], yLabel="Weekly Cashing sum(5e9)") 
        X_ = np.linspace(1, len(y_pred_future), len(y_pred_future))
        self.__plotSingleImage(X_, [y_pred_future], title='Prediction for future 13 weeks(91days)', labels=["Prediction"], yLabel='Weekly Cashing sum(5e9)')
         
    def __history_half_shift_plot(self, history, shifted_column_names=[], figSize=(8,5)):
        ndf = pd.DataFrame(history.history)
        plt.figure(figsize=figSize)
        #shift by half epoch to the left for training curve
        for column_name in shifted_column_names:
            ndf[column_name].index -= 0.5
        #index starts from 0, shift 1
        for column_name in ndf.columns:
            ndf[column_name].index += 1.0
            ndf[column_name].plot(ax=plt.gca(), label=column_name)
        plt.legend(loc='best')
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.gca().set_xticks(np.arange(0, len(ndf), len(ndf) // figSize[0] + 1))
        plt.gca().set_title("Loss during Training")
        plt.show()    

consumerFeatureList_ = ConsumerConfidenceExtractor(path="./shouhi2_simplified.csv", extractedColumnIndexes=[4,6,8,10,12,14]).getData()
consumerFeatureDailyList = MonthlyFeatureToDailyFeature(monthStartStr="2018-09", monthEndStr="2021-09", monthFeatureList=consumerFeatureList_).convertData()
print(pd.DataFrame(np.array(consumerFeatureList_).T))
print(pd.DataFrame(np.array(consumerFeatureDailyList).T))

#dateFeatureList = DateFeatureGenerator(startDateStr="2020-02-01", endDateStr="2021-09-30").genDateFeatureList()

print("Running")
#Main(dataFrom="../data/cashing_mesh/", isCreatefeatureDict=True, isCreateData=True, dataTo="./")
#Main(dataFrom="../data/cashing_mesh/", labelFrom="./", isCreateData=True, isCreatefeatureDict=False, dataTo="./")

#restart training
#train autoencoder
#Main(dataFrom="./", labelFrom="./", isCreateData=False, isCreatefeatureDict=False, isTimeGANTraining=True, isTimeGANEnabled=True, isAutoEncoderTraining=True, dataTo="./", leftEpochs=5000)
#Main(dataFrom="./", labelFrom="./", extraFeatures=dateFeatureList+consumerFeatureDailyList, isCreateData=False, isCreatefeatureDict=False, isTimeGANTraining=True, isAutoEncoderTraining=True, dataTo="./", leftEpochs=50)
#train TimeGAN
#Main(dataFrom="./", labelFrom="./", extraFeatures=dateFeatureList+consumerFeatureDailyList, isTimeGANEnabled=True, isCreateData=False, isCreatefeatureDict=False, isTimeGANTraining=True, dataTo="./", leftEpochs=50)
#without timeGAN Training
#Main(dataFrom="./", labelFrom="./", extraFeatures=dateFeatureList+consumerFeatureDailyList, isTimeGANEnabled=True, isCreateData=False, isCreatefeatureDict=False, dataTo="./", leftEpochs=50)
#Main(dataFrom="./", labelFrom="./", extraFeatures=consumerFeatureDailyList, isTimeGANEnabled=True, isCreateData=False, isCreatefeatureDict=False, dataTo="./", leftEpochs=500)
Main(dataFrom="./", labelFrom="./", extraFeatures=consumerFeatureDailyList, isTimeGANEnabled=False, isCreateData=False, isCreatefeatureDict=False, dataTo="./", leftEpochs=5000)
#TimeGAN disabled
#Main(dataFrom="./", labelFrom="./", extraFeatures=dateFeatureList+consumerFeatureDailyList, isCreateData=False, isCreatefeatureDict=False, dataTo="./")

#Resume training
#Main(dataFrom="./", isResumeTraining=True, modelFrom="./", leftEpochs=50)
#just evaluation, no training
#Main(dataFrom="./", isResumeTraining=True, modelFrom="./", leftEpochs=0)