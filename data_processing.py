import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import random
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error
import math

# Load Data
my_data1 = genfromtxt('merged_dataset_BearingTest_1.csv', delimiter=',')  # 2156*8
my_data2 = genfromtxt('merged_dataset_BearingTest_2.csv', delimiter=',')  # 984*4
my_data3 = genfromtxt('merged_dataset_BearingTest_3.csv', delimiter=',')  # 6324*4

start_pt1 = 0
t1b11 = my_data1[start_pt1:, 1]
t1b12 = my_data1[start_pt1:, 2]
t1b21 = my_data1[start_pt1:, 3]
t1b22 = my_data1[start_pt1:, 4]
t1b31 = my_data1[start_pt1:, 5]
t1b32 = my_data1[start_pt1:, 6]
t1b41 = my_data1[start_pt1:, 7]
t1b42 = my_data1[start_pt1:, 8]

start_pt2 = 0
t2b1 = my_data2[start_pt2:, 1]
t2b2 = my_data2[start_pt2:, 2]
t2b3 = my_data2[start_pt2:, 3]
t2b4 = my_data2[start_pt2:, 4]

start_pt3 = 0
t3b1 = my_data3[start_pt3:, 1]
t3b2 = my_data3[start_pt3:, 2]
t3b3 = my_data3[start_pt3:, 3]
t3b4 = my_data3[start_pt3:, 4]

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
# t1b31=t1b31.reshape(-1, 1)
# t1b31 = scaler.fit_transform(t1b31)
# t2b1=t2b1.reshape(-1, 1)
# t2b1 = scaler.fit_transform(t2b1)
# t3b3=t3b3.reshape(-1, 1)
# t3b3 = scaler.fit_transform(t3b3)

failed_case0 = t1b32
failed_case1 = t1b31
failed_case2 = t2b1
failed_case3 = t3b3

survive_case1 = t1b11
survive_case2 = t1b12
survive_case3 = t1b21
survive_case0 = t1b22
survive_case4 = t1b41
survive_case5 = t1b42
survive_case6 = t2b2
survive_case7 = t2b3
survive_case8 = t2b4
survive_case9 = t3b1
survive_case10 = t3b2
survive_case11 = t3b4

time1 = np.linspace(0, my_data1.shape[0] / 6, my_data1.shape[0])
time2 = np.linspace(0, my_data2.shape[0] / 6, my_data2.shape[0])
time3 = np.linspace(0, my_data3.shape[0] / 6, my_data3.shape[0])

# plt.plot(time1, survive_case1, 'o', markersize=1.5, label='survive_case1')
# plt.plot(time1, survive_case2, 'o', markersize=1.5, label='survive_case2')
# plt.plot(time1, survive_case3, 'o', markersize=1.5, label='survive_case3')
# plt.plot(time1, survive_case4, 'o', markersize=1.5, label='survive_case4')
# plt.plot(time1, survive_case5, 'o', markersize=1.5, label='survive_case5')
# plt.plot(time2, survive_case6, 'o', markersize=1.5, label='survive_case6')
# plt.plot(time2, survive_case7, 'o', markersize=1.5, label='survive_case7')
# plt.plot(time2, survive_case8, 'o', markersize=1.5, label='survive_case8')
# plt.plot(time3, survive_case9, 'o', markersize=1.5, label='survive_case9')
# plt.plot(time3, survive_case10, 'o', markersize=1.5, label='survive_case10')
# plt.plot(time3, survive_case11, 'o', markersize=1.5, label='survive_case11')
# plt.show()
#
#
# plt.plot(time1, failed_case0, 'ob', markersize=1.5, label='failed_case1')
# plt.plot(time2, failed_case2, 'or', markersize=1.5, label='failed_case2')
# plt.plot(time3, failed_case3, 'og', markersize=1.5, label='failed_case3')
# plt.show()


# plt.plot(time1,failed_case0, 'o', markersize=1.5, label='Experiment1 failed_case 1')
# plt.plot(time1,failed_case1, 'o', markersize=1.5, label='Experiment1 failed_case 2')
# plt.ylim([0, 0.6])
# plt.legend()
# plt.xlabel('Time (hrs)')
# plt.ylabel('RMS')
#
# ax = plt.subplot(3, 2, 3)
# plt.plot(time2,failed_case2, 'o', markersize=1.5, label='Experiment2 failed_case')
# plt.ylim([0, 0.6])
# plt.legend()
# plt.xlabel('Time (hrs)')
# plt.ylabel('RMS')

# ax = plt.subplot(3, 2, 2)
# plt.plot(time1,survive_case0, 'o', markersize=1.5, label='Experiment1 survive_case 1')
# plt.plot(time1,survive_case1, 'o', markersize=1.5, label='Experiment1 survive_case 2')
# plt.plot(time1,survive_case2, 'o', markersize=1.5, label='Experiment1 survive_case 3')
# plt.plot(time1,survive_case3, 'o', markersize=1.5, label='Experiment1 survive_case 4')
# plt.plot(time1,survive_case4, 'o', markersize=1.5, label='Experiment1 survive_case 5')
# plt.plot(time1,survive_case5, 'o', markersize=1.5, label='Experiment1 survive_case 6')
# plt.ylim([0, 0.6])
# plt.legend()
# plt.xlabel('Time (hrs)')
# plt.ylabel('RMS')
#
# ax = plt.subplot(3, 2, 4)
data_length = len(time2)
failed_data = np.empty([data_length])
survive_data = np.empty([data_length])

plt.plot(time2, failed_case2, '-', markersize=1.5, label='RMS in Failed Case')
plt.plot(time2, survive_case6, '-', markersize=1.5, label='RMS in Survived Case')
# plt.plot(time2,survive_case7, 'o', markersize=1.5, label='Experiment2 survive_case 2')
# plt.plot(time2,survive_case8, 'o', markersize=1.5, label='Experiment2 survive_case 3')
# plt.ylim([0, 0.6])
plt.legend()
plt.xlabel('Time (hrs)')
plt.ylabel('RMS')
# plt.savefig('failure-survived.png')
plt.show()

for i in range(data_length):
    failed_data[i] = random.gauss(0, failed_case2[i] ** 2)
    survive_data[i] = random.gauss(0, survive_case6[i] ** 2)

window_size = 50
mean_data_length = len(time2) - window_size
failed_mean_data = np.empty([mean_data_length])
survived_mean_data = np.empty([mean_data_length])

for i in range(mean_data_length):
    failed_mean_data[i] = np.sum(failed_data[i:i + window_size]) / window_size
    survived_mean_data[i] = np.sum(survive_data[i:i + window_size]) / window_size

window_size = 5

p2p_data_length = len(time2) - window_size
failed_p2p_data = np.empty([p2p_data_length])
survived_p2p_data = np.empty([p2p_data_length])

for i in range(p2p_data_length):
    failed_p2p_data[i] = max(failed_data[i:i + window_size]) - min(failed_data[i:i + window_size])
    survived_p2p_data[i] = max(survive_data[i:i + window_size]) - min(survive_data[i:i + window_size])

figure, axes = plt.subplots(nrows=2, ncols=2)
plt.subplot(2, 2, 1)
plt.plot(time2, failed_data, label='raw failed data')
plt.plot(time2, survive_data, label='raw survived data')
plt.xlabel('Time (hrs)')
plt.ylabel('raw vibration signals')
plt.title('Raw Data')
plt.legend(loc='lower left')

plt.subplot(2, 2, 2)
plt.plot(time2, failed_case2, '-', markersize=1.5, label='raw failed data')
plt.plot(time2, survive_case6, '-', markersize=1.5, label='raw survived data')
plt.xlabel('Time (hrs)')
plt.ylabel('RMS')
plt.title('RMS')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(time2[:len(time2) - 50], failed_mean_data, label='raw failed data')
plt.plot(time2[:len(time2) - 50], survived_mean_data, label='raw survived data')
plt.xlabel('Time (hrs)')
plt.ylabel('mean')
plt.title('mean')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(time2[:len(time2) - window_size], failed_p2p_data, label='raw failed data')
plt.plot(time2[:len(time2) - window_size], survived_p2p_data, label='raw survived data')
plt.xlabel('Time (hrs)')
plt.ylabel('peak to peak')
plt.title('peak to peak')
plt.legend()
figure.tight_layout(pad=0.5)

plt.savefig('features.png')
plt.show()
