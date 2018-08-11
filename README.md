Use Python to reappear the epsilon-SVR function in LibSVM. The svr.py does not
contain shrink function in LibSVM. The svr_shrinking.py adds shrink function to
the svr.py (however, some functions may still be problematic).

# Usage
```python
import numpy as np
import svr

"""
dataMatIn is your input matrix, and labelMat is your labels.
dataMatTra is your test samples.
"""
x_tra = np.mat(dataMatIn)
y_tra = np.mat(labelMat)
x_pre = np.mat(dataMatTra)

clf = svr.svm_train(x_tra, y_tra, 0.1, 150)
y_pre = []
for i in range(x_tra.shape[0]):
    y_pre.append(svm.svm_predict(clf, x_pre[i,:])) # The predict function can only predict a set of data.
```

# Example
The artificial function is taken as an example to test the algorithm.
```python
import svr
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

x_tra = np.arange(0, 20, 0.1)
y_tra = np.sin(x_tra) + np.cos(x_tra) * x_tra

x_tra = np.mat(x_tra).reshape(-1,1)
y_tra = np.mat(y_tra).reshape(-1,1)

# sklearn.svm.SVR
clf_l = svm.SVR(C=150)
clf_l.fit(x_tra, y_tra)
y_pre_l = clf_l.predict(x_tra)
rmse_l = Error(y_tra, y_pre_l)

# our svr function
clf_m = svr.svm_train(x_tra, y_tra, epsilon_svr=0.1, C=150)
y_pre_m = []
for i in x_tra:
    y_pre_m.append(float(svr.svm_predict(clf_m, i)))
rmse_m = Error(y_tra, y_pre_m)

plt.plot(x_tra, y_tra)
plt.plot(x_tra, y_pre_l, '--')
plt.plot(x_tra, y_pre_m, '--')
plt.legend(['Real curve','sklearn.svm.SVR: RMSE='+str(rmse_l),'svr: RMSE='+str(rmse_m)])
plt.show()
```
**The results of the program are as follows:**
Gmax+Gmax2:  [[ 0.00086194]]
Gmax+Gmax2:  [[ 0.00086194]]
****break****
iter:  2458
 =================
obj =  [[-2231.71390416]] , rho =  [[ 2.54547925]]
nSV= 17 	 nBSV= 2
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/PNPNPCNPhard/1.jpg)
