import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from ModelUtil import precision, recall, f1
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['CUDA_VISIBLE_DEVICES']='0'

# cifer 100
INPUT_DIM = 100
MODEL_PATH = '/home/yuchen/CH_mnist/nn_attack_cifer_2d.hdf5'
TRAIN_CSV = '/home/bo/Project/Mixup_demo/Cifar100_Results/MMD_ConfidenceScore_80.csv'
TEST_CSV = '/home/bo/Project/Mixup_demo/Cifar100_Results/MMD_ConfidenceScore_80.csv'
A = 100
B = 101

# # eye
# INPUT_DIM = 5
# MODEL_PATH = '/home/yuchen/CH_mnist/apl_differ_2d_res_update.hdf5'
# A=5
# B=6
# TRAIN_CSV = '/home/haolin/project/attack_APL/Final_preprocess_shadowModel_DenseNet121_result.csv'
# TEST_CSV = '/home/haolin/project/attack_APL/APL_preprocess_targetModel_full_synth_predict_result.csv '

# # chminist
# INPUT_DIM = 8
# MODEL_PATH = '/home/yuchen/CH_mnist/nn_attack_chminist_2d.hdf5'
# A = 8
# B = 9
# TEST_CSV = '/home/yuchen/CH_mnist/chminist_targetMemTf2.csv'

# # location
# INPUT_DIM = 30
# MODEL_PATH = '/home/yuchen/CH_mnist/nn_attack_location_2d.hdf5'
# A=30
# B=31
# TEST_CSV = '/home/bo/Project/Location/target_ConScoreMem.csv'


def getTrainTest(path,A,B,INPUT_DIM):

    '''
    We output our confidence result as an csv file. For example, Eyepacs have 5 classifications, we will get the softmax
    value with 5 columns, and we add the true label to the 6th column, and the member/nonmember label to the 7th column.
    You may check the outputCSV.py

    :param path: .csv file, shadow and target model's output confidence score with true label and member/nonmember label
    :param A: the column number of true label
    :param B: the column number of member/nonmember label
    :param INPUT_DIM: the number of data's classifications(eg:EyePacs will be 5)
    :return: x is [true_label'confidences core, predicted_confidence score], y is member/nonmember label
    '''

    df = pd.read_csv(path)
    X = np.array(df.iloc[:,0:INPUT_DIM])
    max_X = np.sort(X)[:, -1]
    numSample, dim = df.shape
    dataSet = np.empty(df.shape)
    for i in range(numSample):
        dataSet[i] = df.iloc[i].values
    new_data = np.empty((dataSet.shape[0], 4))
    for i in range(dataSet.shape[0]):
        level_number = int(dataSet[i][A])
        new_data[i][0] = dataSet[i][level_number]
        new_data[i][1] = level_number
        new_data[i][2] = dataSet[i][B]
        new_data[i][3] = max_X[i]
    x = new_data[:,[0,3]]
    y = new_data[:,[2]]

    return x, y


def attackerModel(input_dim=2):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def train(x_train, y_train):
    model = attackerModel()
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy', precision, recall, f1])
    model.summary()
    checkpoint = ModelCheckpoint(MODEL_PATH,
                                 monitor='accuracy',
                                 verbose=1, save_best_only=True,
                                 mode='max')
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=32, callbacks=[checkpoint])

def evaluate_attack(x_test, y_test):
    model = load_model(MODEL_PATH,
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy', precision, recall, f1])
    loss, accuracy, Precision, Recall, F1 = model.evaluate(x_test, y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f' % (loss, accuracy, Precision, Recall, F1))



#x_train,y_train = getTrainTest(TRAIN_CSV,A,B,INPUT_DIM)
x_test,y_test = getTrainTest(TEST_CSV,A,B,INPUT_DIM)
#train(x_train, y_train)
evaluate_attack(x_test, y_test)

