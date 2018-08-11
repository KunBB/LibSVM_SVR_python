Use Python to reappear the epsilon-SVR function in LibSVM. The svr.py does not
contain shrink function in LibSVM. The svr_shrinking.py adds shrink function to
the svr.py (however, some functions may still be problematic).

* Usage
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
for i in range():
    y_pre.append(svm.svm_predict(clf, x_pre[i,:])) # The predict function can only predict a set of data.
```
