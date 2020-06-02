from keras.models import load_model
import pandas as pd

import numpy as np
import math
from PIL import Image
import os
from tqdm import tqdm
from ModelUtil import precision, recall, f1

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


INPUT_DIM = 5
CSV_PATH = '/home/haolin/project/attack_APL/Final_preprocess_targetModel_baseline_predict_result.csv'
A = 5
B = 6
SIZE = 224


def preprocess_image(image_path, desired_size=SIZE):
    """
    Resize the picture to the desired size
    :param image_path: the path of image folder
    :param desired_size: the size that image will be cropped as. The default size is 224*224
    :return: the cropped image
    """
    im = Image.open(image_path)
    im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)

    return im

def set_data(img_path, dataframe):
    """
    Correspond the image to the label and return them.
    :param img_path: the path of images' folder
    :param dataframe: the .csv file that shows relation between image and label
    :return: Image, Label and the name of Image
    """
    N = len(os.listdir(img_path))
    x_ = np.empty((N, SIZE, SIZE, 3), dtype=np.uint8)
    y_ = np.empty(N)
    image_names = np.empty(N, dtype=np.dtype(('U', 15)))
    for i, img_name in enumerate(tqdm(os.listdir(img_path))):
        x_[i, :, :, :] = preprocess_image(img_path + img_name)
        y_[i] = dataframe.loc[img_name.split('.')[0], 'level']
        image_names[i] = img_name

    return x_, y_

def getAvgLoss(shadow_model_path, train_imgpath,train_labelpath):
    '''
      get the average loss value of shadow model
      :param shadow_model_path: confidence score vector with real level and lable
      :param train_labelpath: csv file of training label
      :param train_imgpath: folder of training image
      :return: average loss value
      '''
    model = load_model(shadow_model_path,
                       custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    x_train, y = set_data(train_imgpath, train_labelpath)
    y = pd.get_dummies(y).values
    loss_avg = model.evaluate(x_train, y)[0]
    print(loss_avg)

    return loss_avg

model_path = '/home/bo/Project/densenet.hdf5'
train_img_path = '/home/bo/Project/Eyes_data/second_train/'
label_df = pd.read_csv('/home/bo/Project/Eyes_data/second_label.csv', error_bad_lines=False, index_col=0)
loss_avg = getAvgLoss(model_path, train_img_path, label_df)

def getData():
    data = pd.read_csv(CSV_PATH)
    numSample, dim = data.shape
    dataSet = np.empty(data.shape)
    for i in range(numSample):
        dataSet[i] = data.iloc[i].values
    new_data = np.empty((dataSet.shape[0], 3))
    for i in range(dataSet.shape[0]):
        level_number = int(dataSet[i][A])
        new_data[i][0] = dataSet[i][level_number]
        new_data[i][1] = level_number
        new_data[i][2] = dataSet[i][B]
    return new_data


def Attack():

    data = getData()
    accurate_sample_train = 0
    accurate_sample_test = 0
    for i in range(data.shape[0]):
        loss= - math.log(data[i][0])
        if loss<=loss_avg:
            if data[i][2]==1:
                accurate_sample_train+=1
        if loss>loss_avg:
            if data[i][2]==0:
                accurate_sample_test+=1

    accuracy=(accurate_sample_train + accurate_sample_test)/data.shape[0]
    print(accuracy)
    return accuracy


Attack()








