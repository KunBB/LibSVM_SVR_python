Use Python to reappear the epsilon-SVR function in LibSVM. The svr.py does not
contain shrink function in LibSVM. The svr_shrinking.py adds shrink function to
the svr.py (however, some functions may still be problematic).  

The detailed analysis of the SVR program in LIBSVM could be seen in https://xuyunkun.com/2018/01/30/libsvm%20SVR%E9%83%A8%E5%88%86%E8%AF%A6%E8%A7%A3/#more.  


# Dependencies
numpy

# Usage
* Train:
```python
clf = svr.svm_train(dataMatIn, labelMat, epsilon_svr, C, eps=1e-3, kernel_type='rbf', degree=3, gamma=0.1, coef0=0.0)
```  
|Parameter|Interpretation|
|:-:|:-|
|dataMatIn|Training vectors|
|labelMat|Target values|
|epsilon_svr|The interval width parameter in SVR|
|C|Penalty parameter C of the error term|
|eps|Tolerance for stopping criterion|
|kernel_type|Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'laplacian' or 'morlet'. |
|degree|Kernel coefficient for 'poly'|
|gamma|Kernel coefficient for 'rbf', 'poly', 'sigmoid' and 'laplacian'|
|coef0|Kernel coefficient for 'poly' and 'sigmoid'|

* Predict
```python
y_pre = []
for i in range(x_tra.shape[0]):
    y_pre.append(svm.svm_predict(clf, x_pre[i,:])) # The predict function can only predict a set of data.
```


# Example
The artificial function is taken as an example to test the algorithm.
```python
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
for i in range(100):
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

plt.scatter(x_tra.tolist(),y_tra.tolist(),c='w',marker='^',edgecolors='g',linewidths=2)
plt.plot(x_tes, y_tes)
plt.plot(x_tes, y_pre_l, '--')
plt.plot(x_tes, y_pre_m, '--')
plt.legend(['Training samples','Real curve','sklearn.svm.SVR: RMSE='+str(rmse_l),'svr: RMSE='+str(rmse_m)])
plt.show()
```
**The results of the program are as follows:**  
Gmax+Gmax2:  [[ 0.0008514]]
Gmax+Gmax2:  [[ 0.0008514]]
\*\*\*\*break\*\*\*\*
iter:  5004
 \=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=
obj =  [[-6272.41495558]] , rho =  [[-3.09014393]]
nSV= 21 	 nBSV= 1
![Loading...](https://raw.githubusercontent.com/KunBB/LibSVM_SVR_python/master/Example/Figure_1.png)
