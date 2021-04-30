from pandas import read_csv
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# load the dataset
test1 = read_csv('merged_dataset_BearingTest_1.csv')
test2 = read_csv('merged_dataset_BearingTest_2.csv')
test3 = read_csv('merged_dataset_BearingTest_3.csv')

data131 = np.array(test1['Bearing 3 C1'].values)
data132 = np.array(test1['Bearing 3 C2'].values)
data21 = np.array(test2['Bearing 1'].values)
data33 = np.array(test3['Bearing 3'].values)

# train and test
test1 = data132[200:]
test2 = data21
test3 = data33

# Time
N1 = test1.shape[0]
time1 = np.linspace(0, N1/6, N1)
N2 = test2.shape[0]
time2 = np.linspace(0, N2/6, N2)
N3 = test3.shape[0]
time3 = np.linspace(0, N3/6, N3)

######################## 1
MSE = 0
i = 1
difference = 0
while difference <= 0.012:
    linear_reg = LinearRegression().fit(time1.reshape(-1, 1)[0:i], test1[0:i])
    difference = test1[i+1] - linear_reg.predict(np.array([[time1[i+1]]]))
    i = i + 1

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

linear_regression = linear_reg.predict(time1[0:i+1].reshape(-1,1))

param1, param_cov1 = curve_fit(exponential, time1[0:len(time1)-i-15], test1[i:len(test1)-15], p0=[1, 0.6, 0.3], maxfev=2000)
ans1 = param1[0]*(np.exp(param1[1]*time1[0:len(time1)-i+5])) + param1[2]

fig, axs = plt.subplots(3, figsize=(12,10))
fig.text(0.1, 0.1, 'RMS', va='center', rotation='vertical', fontsize=14)

axs[0].plot(time1[280*6:i+1], test1[280*6:i+1], '.', color='purple', markersize=5, label = 'Real Data')
axs[0].plot(time1[280*6:i+1], linear_regression[280*6:], '-', color='cyan', linewidth=2, label='Linear Prediction')
axs[0].plot(time1[i:], test1[i:], '.', color='purple', markersize=5)
axs[0].plot(time1[280*6:], 0.3*np.ones_like(time1[280*6:]), '--')

new_t1 = np.linspace(0, (N1+5)/6, N1+5)
axs[0].plot(new_t1[i:-20], ans1[:-20], '-', color='chartreuse', linewidth=2, label='Exponential Train')
axs[0].plot(new_t1[-21:], ans1[-21:], '-', color='tomato', linewidth=2, label='Exponential Prediction')

####### Confidence region
ci = 0.95
pp = (1. + ci) / 2.
nstd = stats.norm.ppf(pp)

exponential_region = time1[0:len(time1)-i+30]

perr = np.sqrt(np.diag(param_cov1))
popt_up = param1 + nstd * perr
popt_dw = param1 - nstd * perr

exponential_region_prediction_up = popt_up[0]*(np.exp(popt_up[1]*exponential_region)) + popt_up[2]
exponential_region_prediction_down = popt_dw[0]*(np.exp(popt_dw[1]*exponential_region)) + popt_dw[2]

new_t1 = np.linspace(0, (N1+30)/6, N1+30)
axs[0].plot(new_t1[i:-57], exponential_region_prediction_up[:-57], '--', color='red', label='95% Confidence Interval')
axs[0].plot(new_t1[i:], exponential_region_prediction_down, '--', color='red')

axs[0].legend()
axs[0].set_title('Test 1 Bearing 3 Channel 2', fontsize=11)

###################### 2
MSE = 0
i = 1
difference = 0
while difference <= 0.012:
    linear_reg = LinearRegression().fit(time2.reshape(-1, 1)[0:i], test2[0:i])
    difference = test2[i+1] - linear_reg.predict(np.array([[time2[i+1]]]))
    i = i + 1

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

linear_regression = linear_reg.predict(test2[0:i+1].reshape(-1,1))

param2, param_cov2 = curve_fit(exponential, time2[0:len(time2)-i-15], test2[i:len(test2)-15], p0=[1, 0.6, 0.3], maxfev=2000)
ans2 = param2[0]*(np.exp(param2[1]*time2[0:len(time2)-i+5])) + param2[2]

axs[1].plot(time2[480:i+1], test2[480:i+1], '.', color='purple',markersize=5, label = 'Linear Real')
axs[1].plot(time2[480:i+1], linear_regression[480:], 'c-', linewidth = 2, label = 'Linear Predict')
axs[1].plot(time2[i:], test2[i:], '.', color='purple',markersize=5, label = 'Exponential Real')
axs[1].plot(time2[480:], 0.3*np.ones_like(time2[480:]), '--')

new_t2 = np.linspace(0, (N2+5)/6, N2+5)
axs[1].plot(new_t2[i:-20], ans2[:-20], '-', color='chartreuse',linewidth = 2, label = 'Exponential Train')
axs[1].plot(new_t2[-21:], ans2[-21:], '-', color='tomato', linewidth=2, label='Exponential Predict')

####### Confidence region
ci = 0.95
pp = (1. + ci) / 2.
nstd = stats.norm.ppf(pp)

exponential_region = time2[0:len(time2)-i+15]

perr = np.sqrt(np.diag(param_cov2))
popt_up = param2 + nstd * perr
popt_dw = param2 - nstd * perr

exponential_region_prediction_up = popt_up[0]*(np.exp(popt_up[1]*exponential_region)) + popt_up[2]
exponential_region_prediction_down = 1e-16*(np.exp(popt_dw[1]*exponential_region)) + popt_dw[2]

new_t2 = np.linspace(0, (N2+15)/6, N2+15)
axs[1].plot(new_t2[i:-50], exponential_region_prediction_up[:-50], '--', color='red', label='95% Confidence Interval')
axs[1].plot(new_t2[i:], exponential_region_prediction_down, '--', color='red')

axs[1].set_title('Test 2 Bearing 1', fontsize=11)

####################### 3
MSE = 0
i = 1
difference = 0
while difference <= 0.012:
    linear_reg = LinearRegression().fit(time3.reshape(-1, 1)[0:i], test3[0:i])
    difference = test3[i+1] - linear_reg.predict(np.array([[time3[i+1]]]))
    i = i + 1

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

linear_regression = linear_reg.predict(test3[0:i+1].reshape(-1,1))

param3, param_cov3 = curve_fit(exponential, time3[0:len(time3)-i-15], test3[i:len(test3)-15], p0=[1, 0.6, 0.3], maxfev=2000)
ans3 = param3[0]*(np.exp(param3[1]*time3[0:len(time3)-i+5])) + param3[2]

axs[2].plot(time3[980*6:i+1], test3[980*6:i+1], '.', color='purple',markersize=5, label = 'Linear Real')
axs[2].plot(time3[980*6:i+1], linear_regression[980*6:], 'c-', linewidth = 2, label = 'Linear Predict')
axs[2].plot(time3[i:], test3[i:], '.', color='purple',markersize=5, label = 'Exponential Real')
axs[2].plot(time3[980*6:], 0.3*np.ones_like(time3[980*6:]), '--')

new_t3 = np.linspace(0, (N3+5)/6, N3+5)
axs[2].plot(new_t3[i:-20], ans3[:-20], '-', color='chartreuse', linewidth = 2, label = 'Exponential Train')
axs[2].plot(new_t3[-21:], ans3[-21:], '-', color='red',linewidth=2, label='Exponential Predict')

####### Confidence region
ci = 0.95
pp = (1. + ci) / 2.
nstd = stats.norm.ppf(pp)

exponential_region = time3[0:len(time3)-i]

perr = np.sqrt(np.diag(param_cov3))
popt_up = param3 + nstd * perr
popt_dw = param3 - nstd * perr

exponential_region_prediction_up = popt_up[0]*(np.exp(popt_up[1]*exponential_region)) + popt_up[2]
exponential_region_prediction_down = popt_dw[0]*(np.exp(popt_dw[1]*exponential_region)) + popt_dw[2]

axs[2].plot(time3[i:-45], exponential_region_prediction_up[:-45], '--', color='red', label='upper confidence interval')
axs[2].plot(time3[i:], exponential_region_prediction_down, '--', color='red', label='lower confidence interval')

axs[2].set_xlabel('Time [hr]', fontsize=14)
axs[2].set_title('Test 3 Bearing 3', fontsize=11)
plt.show()

# plt.savefig('Combined_zoom.png')