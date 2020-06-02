import pandas as pd
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Eye
INPUT_DIM = 5
MODEL_PATH = '/home/bo/Project/densenet.hdf5'
CSV_PATH = '/home/bo/Project/MemguardData/target_eye_all_ori.csv'
A = 5
B = 6


def getData():

    data = pd.read_csv(CSV_PATH)
    X = np.array(data.iloc[:,0:INPUT_DIM])
    max_X = np.argmax(X, axis = 1)
    numSample, dim = data.shape
    dataSet = np.empty(data.shape)

    for i in range(numSample):
        dataSet[i] = data.iloc[i].values
    new_data = np.empty((dataSet.shape[0], 3))
    for i in range(dataSet.shape[0]):
        level_number = int(dataSet[i][A]) #  A: the column number of true label
        new_data[i][0]= max_X[i]
        new_data[i][1]=level_number
        new_data[i][2]=dataSet[i][B]  #B:the column number of member/nonmember label

    return new_data


def attack():

    data = getData()
    accurate_sample_train = 0
    accurate_sample_test = 0
    for i in range(data.shape[0]):
        if data[i][0]==data[i][1]:
            if data[i][2]==1:
                accurate_sample_train+=1

        else:
            if data[i][2]==0:
                accurate_sample_test+=1

    accuracy=(accurate_sample_train + accurate_sample_test)/data.shape[0]
    print(accuracy)
    return accuracy

attack()
