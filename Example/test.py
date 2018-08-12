import svr
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_squared_error

def Error(y_predicted, y_test):
    '''
    Calculate the root mean squart error between the true values and predicted
    values.
    '''
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    return rmse

x_tra = []
for i in range(50):
    x_tra.append(random.uniform(0,30))
x_tra.sort()
y_tra = np.sin(x_tra) + np.cos(x_tra) * x_tra
x_tra = np.mat(x_tra).reshape(-1,1)
y_tra = np.mat(y_tra).reshape(-1,1)

x_tes = []
for i in range(1000):
    x_tes.append(random.uniform(0,30))
x_tes.sort()
y_tes = np.sin(x_tes) + np.cos(x_tes) * x_tes
x_tes = np.mat(x_tes).reshape(-1,1)
y_tes = np.mat(y_tes).reshape(-1,1)

# sklearn.svm.SVR
clf_l = svm.SVR(C=150,gamma=0.1)
clf_l.fit(x_tra, y_tra)
y_pre_l = clf_l.predict(x_tes)
rmse_l = Error(y_tes, y_pre_l)

# our svr function
clf_m = svr.svm_train(x_tra, y_tra, epsilon_svr=0.1, C=150)
y_pre_m = []
for i in x_tes:
    y_pre_m.append(float(svr.svm_predict(clf_m, i)))
rmse_m = Error(y_tes, y_pre_m)

plt.scatter(x_tra.tolist(),y_tra.tolist(),c='w',marker='^',edgecolors='r',linewidths=2)
plt.plot(x_tes, y_tes)
plt.plot(x_tes, y_pre_l, '--')
plt.plot(x_tes, y_pre_m, '--')
plt.legend(['Real curve','sklearn.svm.SVR: RMSE='+str(rmse_l),'svr: RMSE='+str(rmse_m),'Training samples'])
plt.show()