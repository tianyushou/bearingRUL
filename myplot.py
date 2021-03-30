import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import L1L2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time #helper libraries

# ------------------------------- #
# For Plotting graphs for report  #
# ------------------------------- #

test = np.load('PlotTestT3B3.npy')
pred0 = np.load('PlotPredictT3B3_l80_p70.npy')
pred1 = np.load('PlotPredictT3B3_l20_p70.npy')
pred2 = np.load('PlotPredictT3B3_l20_p60.npy')

test3 = np.load('PlotTestT1B3.npy')
pred3 = np.load('PlotPredictT1B3_l20_p60.npy')

test4 = np.load('PlotTestT2B1.npy')
pred4 = np.load('PlotPredictT2B1_l20_p60.npy')
# plt.plot(test)
# plt.plot(pred2)
# plt.show()
timex0 = test.shape[0]
timex0 = np.linspace(0,(timex0-1)*10,timex0)
timex0 /= 60

timex3 = test3.shape[0]
timex3 = np.linspace(0,(timex3-1)*10,timex3)
timex3 /= 60

timex4 = test4.shape[0]
timex4 = np.linspace(0,(timex4-1)*10,timex4)
timex4 /= 60

fig, axs = plt.subplots(3, 1)
axs[0].plot(timex0, test, label='test curve')
axs[0].plot(timex0, pred0, label = 'memorized time length = 13.3 hrs, predicted time length = 11.7 hrs')

axs[0].legend()
axs[0].set_title('Trained with bearing 1 and 2, test with bearing 3')

axs[1].plot(timex0, test, label='test curve')
axs[1].plot(timex0, pred1, label = 'memorized time length = 3.3 hrs, predicted time length = 11.7 hrs')

axs[1].legend()
# axs[1].set_title('input size = 20, output size = 70')

axs[2].plot(timex0, test, label='test curve')
axs[2].plot(timex0, pred2, label = 'memorized time length = 3.3 hrs, predicted time length = 10 hrs')

axs[2].legend()
# axs[2].set_title('input size = 20, output size = 60')

for ax in axs.flat:
    ax.set(xlabel='Time (hr)', ylabel='RMS')

for ax in axs.flat:
    ax.label_outer()

plt.show()


fig, axs = plt.subplots(3, 1)
axs[0].plot(timex0, test, label='test curve')
axs[0].plot(timex0, pred2, label = 'memorized time length = 3.3 hrs, predicted time length = 10 hrs')

axs[0].legend()
axs[0].set(xlabel='Time (hr)', ylabel='RMS')
axs[0].set_title('Trained with bearing 1 and 2, test with bearing 3')

axs[1].plot(timex3, test3, label='test curve')
axs[1].plot(timex3, pred3, label = 'memorized time length = 3.3 hrs, predicted time length = 10 hrs')

axs[1].legend()
axs[1].set(xlabel='Time (hr)', ylabel='RMS')
axs[1].set_title('Trained with bearing 1 and 3, test with bearing 2')

axs[2].plot(timex4, test4, label='test curve')
axs[2].plot(timex4, pred4, label = 'memorized time length = 3.3 hrs, predicted time length = 10 hrs')

axs[2].legend()
axs[2].set_title('Trained with bearing 2 and 3, test with bearing 1')
axs[2].set(xlabel='Time (hr)', ylabel='RMS')

plt.tight_layout()

# for ax in axs.flat:
#     ax.set(xlabel='Time (hr)', ylabel='RMS')

# for ax in axs.flat:
#     ax.label_outer()

plt.show()


# t = np.load('PlotTestT2B1.npy')
# p = np.load('rawpredictT2B1.npy')
# look_back = 20
# predict_len = 60
# print(t.shape,'   ',p.shape)
# testPredictPlot = np.zeros((t.shape[0] + predict_len,1))
# testPredictPlot[:, :] = np.nan
# testPredictPlot[-len(p):, :] = p.reshape((len(p),1))
# plt.plot(t)
# plt.plot(testPredictPlot)
# plt.show()