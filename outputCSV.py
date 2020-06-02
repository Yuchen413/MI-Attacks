
import tqdm
from keras.models import load_model
from ModelUtil import precision, recall, f1
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES']='1'


model_path = '/home/bo/Project/densenet.hdf5'
train_img_path = '/home/bo/Project/Eyes_data/first_train/'
test_img_path = '/home/bo/Project/Eyes_data/first_test/'
label_df = pd.read_csv('/home/bo/Project/Eyes_data/first_label.csv', error_bad_lines=False, index_col=0)

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


def predict(X):
    model = load_model(model_path,
                     custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    ret = model.predict(X)

    return ret

def ouput_csv(X_, Y_, csv_path):
    model = load_model(model_path,
                       custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    data = model.predict(X_)
    dataDF = pd.DataFrame(data)
    dataDF['level'] = Y_[:, 0]
    dataDF['label'] = Y_[:, 1]
    print(dataDF)
    dataDF.to_csv(csv_path, index=False)


x_train, y_train =  set_data(train_img_path,label_df)
y_in = np.c_[y_train, np.ones(y_train.shape[0])]

x_test, y_test = set_data(test_img_path,label_df)
y_out = np.c_[y_test, np.zeros(y_test.shape[0])]

X_ = np.r_[x_train, x_test]
Y_ = np.r_[y_in, y_out]

ouput_csv(X_, Y_, 'target or shadow output.csv')