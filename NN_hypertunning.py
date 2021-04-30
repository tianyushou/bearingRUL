from tensorflow import keras
from keras.models import Sequential, load_model
from tensorflow.keras import datasets, layers, models
from keras.layers import LSTM, Bidirectional
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import kerastuner as kt

look_back = 40
predict_len = 60

# get training and test data
def get_data():
	bearing_test_1 = read_csv('merged_dataset_BearingTest_1.csv')
	bearing_test_2 = read_csv('merged_dataset_BearingTest_2.csv')
	bearing_test_3 = read_csv('merged_dataset_BearingTest_3.csv')


	t1_b3_c1 = np.array(bearing_test_1['Bearing 3 C1'].values)
	t1_b3_c2 = np.array(bearing_test_1['Bearing 3 C2'].values)
	t3_b3 = np.array(bearing_test_3['Bearing 3'].values)

	t2_b1 = np.array(bearing_test_2['Bearing 1'].values)

	return t1_b3_c1, t1_b3_c2, t3_b3, t2_b1

def get_ptp_data():
	case1 = pd.read_excel('case1.xlsx',engine='openpyxl')
	case2 = pd.read_excel('case2.xlsx',engine='openpyxl')
	case3 = pd.read_excel('case3.xlsx',engine='openpyxl')
	case4 = pd.read_excel('case4.xlsx',engine='openpyxl')

	case1 = np.array(list(case1.iloc[:,0]))
	case2 = np.array(list(case2.iloc[:,0]))
	case3 = np.array(list(case3.iloc[:,0]))
	case4 = np.array(list(case4.iloc[:,0]))

	return case2,case3, case4, case4

	


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, future_size = 1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - future_size -1):
		a = dataset[i:(i+look_back)]
		b = dataset[i+look_back: i+look_back + future_size]
		dataX.append(a)
		dataY.append(b)
	return np.array(dataX), np.array(dataY)

def create_test_dataset(dataset, look_back=1, future_size = 1):
	dataX = []
	for i in range(len(dataset) - look_back -1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
	return np.array(dataX)

# Function for tunner to build the model
def model_builder(hp):
	model = keras.Sequential()
	# Tune the drop out probability for the whole mdel
	drop = hp.Float('dropout', 0.0, 0.3, step=0.1, default=0.0)
	# Tune the number of 1D convolution layers
	for i in range(hp.Int('Conv_layers',0,3)):
		## Tune the number of filter size
		filters = hp.Int('filters_'+str(i),16,112,step=16)
		kernels = hp.Int('kernels_'+str(i),4,16,step=4)
		## CNN layer(s) with tunable parameters
		model.add(layers.Conv1D(filters=filters, kernel_size=kernels, activation='relu', strides=1, padding="same", ))
		## Max polling layer after each Conv layer
		model.add(layers.MaxPooling1D(pool_size=2, padding="same"))
	
	# Tune the number of channels
	channels = hp.Int('Channel',1,3)
	sequence_len = hp.Int('Sequence_length',10,look_back,step=10)
	## Fully connected layer between CNN and LSTM
	model.add(layers.Dense(channels * sequence_len, activation='relu'))
	model.add(layers.Dropout(drop))

	## Tune the number of bidirectional LSTM
	for i in range(hp.Int('Bidirectional_LSTM_layers',0,2)):
		multiple = hp.Int('channel_multiple_'+str(i),4,16,step = 4)
		## Bidirectional LSTM layer(s)
		model.add(Bidirectional(LSTM(channels * multiple, return_sequences=True)))
		model.add(layers.Wrapper(layers.Dropout(drop)))
	
	## Tune the cell size for LSTM
	cell_size = hp.Int('LSTM_cell_size',look_back,look_back+160,step = 40)
	model.add(LSTM(cell_size))
	model.add(layers.Wrapper(layers.Dropout(drop)))

	## Tune the number of FC layer before the final output layer
	for i in range(hp.Int('FC_layers',0,2)):
		fc_unit = hp.Int('FC_unit_'+str(i),40,320, step=40)
		model.add(layers.Dense(fc_unit, activation='relu'))
		model.add(layers.Dropout(drop))

	## Final output linear FC layer
	model.add(layers.Dense(predict_len, activation='linear'))

	# Tune the learning rate as well
	lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])


	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=lr), 
		loss='mse', 
		metrics=['mse']
	)

	return model


# Get data
train_feature1, train_feature2, train_feature3, test = get_ptp_data()

print('initial data shape: ',train_feature1.shape)

# create trainning and test data
X_train1, y_train1 = create_dataset(train_feature1, look_back, predict_len)
X_train2, y_train2 = create_dataset(train_feature2,  look_back, predict_len)
X_train3, y_train3 = create_dataset(train_feature3,  look_back, predict_len)
X_validation, Y_validation = create_dataset(test, look_back, predict_len)

X_test = create_test_dataset(test, look_back, predict_len)

X_train = np.append(X_train3,X_train2,0)
Y_train = np.append(y_train3,y_train2,0)





X_train1 = np.reshape(X_train1, (X_train1.shape[0], 1, X_train1.shape[1]))
X_train2 = np.reshape(X_train2, (X_train2.shape[0], 1, X_train2.shape[1]))
X_train3 = np.reshape(X_train3, (X_train3.shape[0], 1, X_train3.shape[1]))
X_validation = np.reshape(X_validation, (X_validation.shape[0],1,X_validation.shape[1]))

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# -----------------------------------------------Tuning------------------------------------------- #
# Create a hyperparameter tunner
tuner = kt.Hyperband(
	model_builder,
	objective='val_loss',
	max_epochs=100,
	factor=3,
	directory='peak2peak',
	project_name='Bearing_RUL_Prediction_LSTM'
)

# stop early when there's no improvement
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# search for the optimal parameter
tuner.search(X_train3, y_train3, epochs=200, validation_data=(X_train1,y_train1), callbacks=[stop_early])

# Get the optimal hyperparameters
best_params = tuner.get_best_hyperparameters(num_trials=1)[0]

opt_conv_layers = best_params.get('Conv_layers')
opt_bilstm_layers = best_params.get('Bidirectional_LSTM_layers')
opt_fc_layers = best_params.get('FC_layers')

param_string = f"""
The hyperparameter search is complete:
\tThe optimal drop out rate = {best_params.get('dropout')},
\tThe optimal number of CNN layer(s) is {best_params.get('Conv_layers')},
\tThe optimal number of filter(s) is {best_params.get('filters_'+str(opt_conv_layers))}
\tThe optimal kernel size is {best_params.get('kernels_'+str(opt_conv_layers))}
\tThe optimal channel size is {best_params.get('Channel')}
\tThe optimal sequence length is {best_params.get('Sequence_length')}
\tThe optimal number of bidirectional LSTM layer(s) is {best_params.get('Bidirectional_LSTM_layers')}
\tThe optimal multiple of channel is {best_params.get('channel_multiple_'+str(opt_bilstm_layers))}
\tThe optimal LSTM cell number is {best_params.get('LSTM_cell_size')}
\tThe optimal number of FC layer(s) is {best_params.get('FC_layers')}
\tThe optimal number of units in FC layer is {best_params.get('FC_unit_'+str(opt_fc_layers))}
\tThe optimal learning rate is {best_params.get('learning_rate')}.
"""
print(param_string)

# Write the best param to file
text_file = open("Best_param.txt", "w")
text_file.write(param_string)
text_file.close()

# Rebuild the best model
best_model = tuner.hypermodel.build(best_params)

# Train our best model
best_model.fit(X_train3, y_train3, epochs=200, validation_data=(X_validation,Y_validation))
# best_model.fit(X_train3, y_train3, epochs=200)
# best_model.fit(X_train1, y_train1, epochs=200, validation_data=(X_validation,Y_validation))
best_model.save('Best_model_peak2peak_train3_valid_train1')

# ----------------------------------------End of Tuning------------------------------------------- #
# best_model = keras.models.load_model("Best_model_fit_train3_valid_train1")

# Test 
test_forecast = np.array(best_model.predict(X_train2))

forecast_len = test_forecast.shape[0]

# Plot last 70 predictions 
for i in range(forecast_len-130,forecast_len):
	orig = np.zeros((test.shape[0] + predict_len,1))
	orig[:,:] = np.nan
	orig[0:test.shape[0],0] = test

	testPredictPlot = np.zeros((test.shape[0] + predict_len,1))
	testPredictPlot[:, :] = np.nan
	testPredictPlot[look_back + i: look_back + i + predict_len, :] = test_forecast[i,:].reshape((len(test_forecast[i]),1))

	num_datapoint = test.shape[0] + predict_len
	time = np.linspace(0.0,(num_datapoint-1)*10.0,num_datapoint)
	time /= 60

	threshold = np.zeros((test.shape[0] + predict_len,1))
	threshold += 0.17

	plt.plot(time, orig, label='test curve')
	plt.plot(time, testPredictPlot, label = 'predicted curve')
	plt.legend()
	plt.plot(time,threshold, '--')
	
	plt.xlabel('Time (hrs)')
	plt.ylabel('Peak to peak value')
	plt.savefig('pic_train2/wholePlot_{}.png'.format(i))
	plt.clf()