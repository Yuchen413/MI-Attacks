from functools import partial
import tensorflow as tf
from keras.datasets import cifar100
import pandas as pd
import numpy as np
from numpy import *
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# #chminist
# INPUT_DIM = 8
# path_nonmem = '/home/bo/Project/CH_MNIST/sha_non_member_forMMD.csv'
# path_mix = '/home/bo/Project/CH_MNIST/ori_ConfidenceScores.csv'
# A = 8
# B = 9

# #cifer100
# INPUT_DIM = 100
# path_nonmem = '/home/bo/Project/Sobel_test/Sha_ConfidenceScores.csv'
# path_mix = '/home/bo/Project/Mixup_demo/fullCifar100_MMDMixUp_Results/MMD_MixUp_Resnet_weights20.csv'
# A = 100
# B = 101

# #mnist
# INPUT_DIM = 10
# path_nonmem = '/home/yuchen/CH_mnist/mnist_ori_sobel.csv'
# path_mix = '/home/yuchen/CH_mnist/mnist_ori.csv'
# A = 10
# B = 11

# eye
INPUT_DIM = 5
A=5
B=6
path_nonmem = '/home/haolin/project/attack_APL/Final_preprocess_shadowModel_ResNet50_result.csv'
path_mix= '/home/haolin/project/attack_APL/Final_preprocess_targetModel_baseline_predict_result.csv'

def compute_pairwise_distances(x, y):
      """Computes the squared pairwise Euclidean distances between x and y.
      Args:
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]
      Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples].
      Raises:
        ValueError: if the inputs do no matched the specified dimensions.
      """

      if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

      if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

      norm = lambda x: tf.reduce_sum(tf.square(x), 1)

      return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
      r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
      We create a sum of multiple gaussian kernels each having a width sigma_i.
      Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        sigmas: a tensor of floats which denote the widths of each of the
          gaussians in the kernel.
      Returns:
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
      """
      beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

      dist = compute_pairwise_distances(x, y)

      s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

      return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):
      r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
      Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
      the distributions of x and y. Here we use the kernel two sample estimate
      using the empirical mean of the two distributions.
      MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                  = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
      where K = <\phi(x), \phi(y)>,
        is the desired kernel function, in this case a radial basis kernel.
      Args:
          x: a tensor of shape [num_samples, num_features]
          y: a tensor of shape [num_samples, num_features]
          kernel: a function which computes the kernel in MMD. Defaults to the
                  GaussianKernelMatrix.
      Returns:
          a scalar denoting the squared maximum mean discrepancy loss.
      """
      with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
      return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
      """Adds a similarity loss term, the MMD between two representations.
      This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
      different Gaussian kernels.
      Args:
        source_samples: a tensor of shape [num_samples, num_features].
        target_samples: a tensor of shape [num_samples, num_features].
        weight: the weight of the MMD loss.
        scope: optional name scope for summary tags.
      Returns:
        a scalar tensor representing the MMD loss value.
      """
      sigmas = [
          1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
          1e3, 1e4, 1e5, 1e6
      ]
      gaussian_kernel = partial(
          gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

      loss_value = maximum_mean_discrepancy(
          source_samples, target_samples, kernel=gaussian_kernel)
      loss_value = tf.maximum(1e-4, loss_value) * weight

      return loss_value

def getData_2_dim(path):
    '''
    :param path: csv file get from outputCSV.py or nonMemberGenerator.py.

    For Mix dataset, you can get the csv using outputCSV.py, by feeding your training set and testing set of target to your target model
    For Nonmember dataset, there are two choice. first, you can get the csv using outputCSV.py, by feeding your testing set
    of shadow model to your shadow model. Second, you may use nonMemberGenerator.py, by feeding any data after sobel or another operator to
    your target model. We found using shadow model was more ideal.

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
    For Nonmember dataset, there are two choice. first, you can get the csv using outputCSV.py, by feeding your testing set
    of shadow model to your shadow model. Second, you may use nonMemberGenerator.py, by feeding any data after sobel or another operator to
    your target model. We found using shadow model was more ideal.

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


def getData_all_dim(path):
    '''
    :param path: csv file get from outputCSV.py or nonMemberGenerator.py.

    For Mix dataset, you can get the csv using outputCSV.py, by feeding your training set and testing set of target to your target model
    For Nonmember dataset, there are two choice. first, you can get the csv using outputCSV.py, by feeding your testing set
    of shadow model to your shadow model. Second, you may use nonMemberGenerator.py, by feeding any data after sobel or another operator to
    your target model. We found using shadow model was more ideal.

    :return: numpy array that we use for attack.
    '''
    data = pd.read_csv(path)
    numSample, dim = data.shape
    dataSet = np.empty(data.shape)
    for i in range(numSample):
          dataSet[i] = data.iloc[i].values
    new_data = np.empty((dataSet.shape[0], dataSet.shape[1] + 1))
    for i in range(dataSet.shape[0]):
          level_number = int(dataSet[i][A])
          new_data[i][0] = dataSet[i][level_number]
          new_data[i][1:new_data.shape[1]] = dataSet[i]
    return new_data


def attackAccuracy_distance():
    # or you may try 3_dim or all_dim
    n_data = getData_2_dim(path_nonmem)
    m_data = getData_2_dim(path_mix)

    non_member = tf.convert_to_tensor(n_data[:, [0, 3]])
    mix = tf.convert_to_tensor(m_data[:, [0, 3]])
    non_member = tf.cast(non_member, tf.float32)
    mix = tf.cast(mix, tf.float32)
    distance = mmd_loss(tf.reshape(non_member, (non_member.shape[0], -1)), tf.reshape(mix, (mix.shape[0], -1)), 1)
    print("Distance is :",distance)

    count_1 = 0
    count_0 = 0

    for i in tqdm(range(mix.shape[0])):
        new_non_member = np.r_[non_member,tf.reshape(mix[i],(1,-1))]
        new_mix = np.delete(mix,i, axis=0)
        new_distance =  mmd_loss(tf.reshape(new_non_member, (new_non_member.shape[0], -1)), tf.reshape(new_mix, (new_mix.shape[0], -1)), 1)
        if new_distance < distance:
            if m_data[i][2] == 1:
                count_1 += 1
        if new_distance >= distance:
            if m_data[i][2] == 0:
                count_0 += 1

    accuracy = (count_0 + count_1) / m_data.shape[0]
    print(accuracy)
    return accuracy


attackAccuracy_distance()