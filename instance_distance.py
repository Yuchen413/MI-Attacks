import pandas as pd
import numpy as np
from numpy import *
from scipy.stats import wasserstein_distance


INPUT_DIM = 8
shadow_path_nonmem = '/home/bo/Project/CH_MNIST/Sob_sha_ConfidenceScores.csv'
shadow_path_mix = '/home/bo/Project/CH_MNIST/sha_ConfidenceScores.csv'
target_path_nonmem = '/home/bo/Project/CH_MNIST/Sob_ori_ConfidenceScores.csv'
target_path_mix = '/home/bo/Project/CH_MNIST/ori_ConfidenceScores.csv'
A = 9
B = 8


def getData_2_dim(path):
    '''
    :param path: csv file get from outputCSV.py or nonMemberGenerator.py.

    For Mix dataset, you can get the csv using outputCSV.py, by feeding your training set and testing set of target to your target model
    For Nonmember dataset, you can use nonMemberGenerator.py, by feedingtraining set and testing set after sobel or another operator to
    your target model.

    :return: numpy array that we use for attack.
    '''
    data = pd.read_csv(path)

    X = data.iloc[:, 0:INPUT_DIM]
    max_X = np.sort(X)[:, -1]
    numSample, dim = data.shape
    dataSet = np.empty(data.shape)
    for i in range(numSample):
        dataSet[i] = data.iloc[i].values
    new_data = np.empty((dataSet.shape[0], 4))
    for i in range(dataSet.shape[0]):
        level_number = int(dataSet[i][A])
        new_data[i][0] = dataSet[i][level_number]
        new_data[i][1] = level_number
        new_data[i][2] = dataSet[i][B]
        new_data[i][3] = max_X[i]
    return new_data


def getData_3_dim(path):
    '''
    :param path: csv file get from outputCSV.py or nonMemberGenerator.py.

    For Mix dataset, you can get the csv using outputCSV.py, by feeding your training set and testing set of target to your target model
    For Nonmember dataset, you may use nonMemberGenerator.py, by feeding any data after sobel or another operator to
    your target model.

    :return: numpy array that we use for attack.
    '''
    data = pd.read_csv(path)
    X = data.iloc[:, 0:INPUT_DIM]
    max_X = np.sort(X)[:, -1]
    second_X = np.sort(X)[:,-2]
    numSample, dim = data.shape
    dataSet = np.empty(data.shape)
    for i in range(numSample):
        dataSet[i] = data.iloc[i].values
    new_data = np.empty((dataSet.shape[0], 4))
    for i in range(dataSet.shape[0]):
        level_number = int(dataSet[i][A])
        new_data[i][0] = dataSet[i][level_number]
        new_data[i][1] = level_number
        new_data[i][2] = dataSet[i][B]
        new_data[i][3] = max_X[i]
        new_data[i][4] = second_X[i]
    return new_data

shadow_n_data = getData_2_dim(shadow_path_nonmem)
shadow_m_data = getData_2_dim(shadow_path_mix)
target_n_data = getData_2_dim(target_path_nonmem)
target_m_data = getData_2_dim(target_path_mix)

def get_distance_threshold(search_time = 500,stepsize = 0.001):
    '''

    :param search_time: seraching times
    :param stepsize: value for every changes
    :return: distance threshold
    '''
    non_member = shadow_n_data[:, [0, 3]]
    non_member_avg = [mean(non_member[:, [0]]), mean(non_member[:, [1]])]
    print("non member vector:",non_member_avg)
    mix = shadow_m_data[:, [0, 3]]

    dis = np.empty(mix.shape[0])
    for i in range(mix.shape[0]):
        '''you may try different distance calculation'''
        # distance = np.linalg.norm(mix[i] - non_member_avg, ord=1) #Manhattan Distance
        # distance = np.linalg.norm(mix[i] - non_member_avg)    #O
        # distance = np.linalg.norm(mix[i] - non_member_avg, ord = np.inf ) #Chebyshev Distance
        distance = wasserstein_distance(mix[i], non_member_avg)
        dis[i] = distance

    mean_train_d = mean(dis[0:2499]) #depending on specific data
    mean_test_d = mean(dis[2500:5000])

    print("shadow_train_distance",mean_train_d)
    print("shadow_test distance",mean_test_d)

    '''the empirical best threshold will around mean_train_d, 
    print out it can help reduce the range of searching threshold.'''

    threshold = []
    acc = []

    for n in range(search_time):
        threshold.append(mean_train_d + n*stepsize)
        threshold.append(mean_train_d - n*stepsize)

    for j in range(0, len(threshold)):
        accurate_sample_train = 0
        accurate_sample_test = 0
        for i in range(0, dis.shape[0]):
            if dis[i] >= (threshold[j]):
                if shadow_m_data[i][2] == 1:
                    accurate_sample_train += 1
            if dis[i] < (threshold[j]):
                if shadow_m_data[i][2] == 0:
                     accurate_sample_test += 1
        accuracy = (accurate_sample_train + accurate_sample_test) / shadow_m_data.shape[0]
        acc.append(accuracy)
    arg = argmax(acc)
    print("Shadow acc",max(acc))
    thresh= threshold[int(arg)]
    return thresh


def attackAccuracy(threshold):
    '''
    :param threshold: out put result of get_distance_threshold()
    :return: attack accuracy
    '''
    non_member = target_n_data[:, [0, 3]]
    non_member_avg = [mean(non_member[:, [0]]), mean(non_member[:, [1]])]
    print("non member vector:",non_member_avg)
    mix = target_m_data[:, [0, 3]]

    dis = np.empty(mix.shape[0])
    for i in range(mix.shape[0]):
        '''you may try different distance calculation'''
        # distance = np.linalg.norm(mix[i] - non_member_avg, ord=1) #Manhattan Distance
        # distance = np.linalg.norm(mix[i] - non_member_avg)    #O
        # distance = np.linalg.norm(mix[i] - non_member_avg, ord = np.inf ) #Chebyshev Distance
        distance = wasserstein_distance(mix[i], non_member_avg)
        dis[i] = distance

    accurate_sample_train = 0
    accurate_sample_test = 0
    for i in range(0, dis.shape[0]):
        if dis[i] >= (threshold):
            if target_m_data[i][2] == 1:
                accurate_sample_train += 1
        if dis[i] < (threshold):
            if target_m_data[i][2] == 0:
                accurate_sample_test += 1
    accuracy = (accurate_sample_train + accurate_sample_test) / target_m_data.shape[0]
    print("acc on target",accuracy)

    return accuracy


threshold = get_distance_threshold()
attackAccuracy(threshold)


