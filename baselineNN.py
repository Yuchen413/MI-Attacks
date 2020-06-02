import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from ModelUtil import precision, recall, f1
from keras.models import load_model
from keras.optimizers import Adam
import random
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ['CUDA_VISIBLE_DEVICES']='0'

# # location
# INPUT_DIM = 30
# MODEL_PATH = '/home/yuchen/CH_mnist/nn_attack_location_30d.hdf5'

# cifer 100
INPUT_DIM = 100
MODEL_PATH = '/home/yuchen/CH_mnist/nn_attack_cifer_100d.hdf5'
TRAIN_CSV = '/home/bo/Project/Sobel_test/Sha_ConfidenceScores.csv'
TEST_CSV = '/home/bo/Project/Mixup_demo/fullCifar100_MMDMixUp_Results/MMD_MixUp_Resnet_weights80.csv'

# # Eye
# INPUT_DIM = 5
# MODEL_PATH = '/home/yuchen/CH_mnist/nn_attack_Eye_5d.hdf5'

# # chminist
# INPUT_DIM = 8
# MODEL_PATH = '/home/yuchen/CH_mnist/nn_attack_chminist_5d_mem.hdf5'

# ##mnist
# INPUT_DIM = 10
# MODEL_PATH = '/home/yuchen/CH_mnist/nn_attack_mnist_10d.hdf5'


def getTrainTest(path, INPUT_DIM):
    '''
    We output our confidence result as an csv file. For example, Eyepacs have 5 classifications, we will get the softmax
    value with 5 columns, and we add the true label to the 6th column, and the member/nonmember label to the 7th column.
    You may check the outputCSV.py

    :param path: .csv file, shadow and target model's output confidence score with true label and member/nonmember label
    :param INPUT_DIM: the number of data's classifications(eg:EyePacs will be 5)
    :return: x is softmax confidence score, y is member/nonmember label
    '''
    df = pd.read_csv(path)
    x = df.iloc[:, range(INPUT_DIM)].values
    y = df.iloc[:, -1].values
    return x, y

def attackerModel():
    model = Sequential()
    model.add(Dense(512, input_dim=INPUT_DIM, activation='relu'))
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
              batch_size=32,
              callbacks=[checkpoint])

def evaluate_attack(x_test, y_test):
    model = load_model(MODEL_PATH,
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy', precision, recall, f1])
    loss, accuracy, Precision, Recall, F1 = model.evaluate(x_test, y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f' % (loss, accuracy, Precision, Recall, F1))


#x_train,y_train = getTrainTest(TRAIN_CSV)
x_test,y_test = getTrainTest(TEST_CSV)
#train(x_train, y_train)
evaluate_attack(x_test,y_test)
