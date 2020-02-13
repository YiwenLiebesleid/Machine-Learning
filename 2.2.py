import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.io as scio
import time

mnist = scio.loadmat("C:\\Users\\HP\\Desktop\\WPI\\19fall\\machine learning\\TeamAssignment\\2\\mnist-original.mat")
data = mnist['data'].transpose()
label = mnist['label'].transpose().ravel()
# data = mnist['data'].transpose()[0:14000]
# label = mnist['label'].transpose().ravel()[0:14000]

X_train, X_test, y_train, y_test = train_test_split(data,label,test_size=1/7,random_state=0)

X_test = X_test[0:1000]
y_test = y_test[0:1000]

logistic_model = LogisticRegression(penalty='l2',solver='sag',multi_class='multinomial')

t_1 = time.clock()
logistic_model.fit(X_train,y_train)
t_2 = time.clock()

dt = t_2 - t_1

y_hat = logistic_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_hat)

print("accuracy = %.4f, %.4f sec used"%(accuracy,dt))
