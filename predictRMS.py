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


# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1, future_size = 1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - future_size -1):
		a = dataset[i:(i+look_back), 0]
		b = dataset[i+look_back: i+look_back + future_size, 0]
		dataX.append(a)
		dataY.append(b)
	return np.array(dataX), np.array(dataY)

# def create_dataset_3(dataset, look_back=1, future_size = 1):
# 	dataX = []
# 	for i in range(len(dataset) - look_back -1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 	return np.array(dataX)

def main():
	# fix random seed for reproducibility
	np.random.seed(5)

	# load the dataset
	df = read_csv('merged_dataset_BearingTest_3.csv')
	dataset = np.array(df['Bearing 3'].values)
	dataset = np.reshape(dataset, (len(dataset),1) )

	df = read_csv('merged_dataset_BearingTest_1.csv')
	dataset3 = np.array(df['Bearing 3 C1'].values)
	dataset2 = np.array(df['Bearing 3 C2'].values)
	dataset3 = np.reshape(dataset3, (len(dataset3),1) )
	dataset2 = np.reshape(dataset2, (len(dataset2),1) )

	df = read_csv('merged_dataset_BearingTest_2.csv')
	test = np.array(df['Bearing 1'].values)
	test = np.reshape(test, (len(test),1) )



	# reshape into Xt = Dataset[t-look_back : t] and Y= Dataset[t: t + predict_len]
	look_back = 20
	predict_len = 60
	trainX, trainY = create_dataset(dataset, look_back, predict_len)
	trainX2, trainY2 = create_dataset(dataset2, look_back, predict_len)
	trainX3, trainY3 = create_dataset(dataset3, look_back, predict_len)
	# trainX4, trainY4 = create_dataset_2(dataset4, look_back, predict_len)
	testX, _ = create_dataset(test, look_back)

	# Merge training data together
	trainX = np.append(trainX,trainX2,0)
	trainX = np.append(trainX,trainX3,0)

	trainY = np.append(trainY, trainY2, 0)
	trainY = np.append(trainY, trainY3, 0)

	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
	Dense1Dim = 160
	LSTMSize = 160
	# np.save('PlotTestT2B1.npy',test)


	# # Initialize regularization
	# regularizer = L1L2(l1=1e-6, l2=0)

	model = Sequential()
	model.add(Dense(Dense1Dim))
	model.add(LSTM(LSTMSize, input_shape=(1, look_back)))
	model.add(Dropout(0.1))
	model.add(Dense(predict_len))
	model.compile(loss='mse', optimizer='adam')
	model.fit(trainX, trainY, epochs=1000, batch_size=240, verbose=1)

	# make predictions
	testPredict = model.predict(testX)
	print('predict Y : ',testPredict.shape)


	# Assemble prediction data for plotting
	predAssembled = np.zeros(testPredict.shape[0] -1 + predict_len)
	predAssembled[0:predict_len] = testPredict[0]
	for i in range(predict_len, len(predAssembled)):
		predAssembled[i] = testPredict[i-predict_len + 1,-1]

	# shift test predictions for plotting
	testPredictPlot = np.empty_like(test)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[-len(predAssembled):, :] = predAssembled.reshape((len(predAssembled),1))

	# np.save('PlotPredictT2B1_l{}_p{}.npy'.format(int(look_back),int(predict_len)),testPredictPlot)

	plt.plot(test, label='test curve')
	plt.plot(testPredictPlot, label = 'predicted curve')
	plt.legend()
	plt.xlabel('Number of samples')
	plt.ylabel('RMS')
	plt.show()


if __name__ == '__main__':
	main()