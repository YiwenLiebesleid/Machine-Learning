import numpy as np
import scipy.io as scio
import random
import time
from sklearn import svm
from sklearn import cross_validation
import copy
from tqdm import tqdm

def shuffledata(traindata,trainlabel):
    before_shuffle = list(zip(traindata, trainlabel))
    random.shuffle(before_shuffle)
    shuffle_data, shuffle_label = zip(*before_shuffle)
    return np.array(shuffle_data), np.array(shuffle_label)

def readdata():
    mnist = scio.loadmat("C:\\Users\\HP\\Desktop\\WPI\\19fall\\machine learning\\TeamAssignment\\2\\mnist-original.mat")
    data = mnist['data'].transpose()
    label = mnist['label'].transpose().ravel()
    #data, label = shuffledata(data, label)
    return np.array(data), np.array(label)

N = 1000

def fit(i,data,label,C,gamma):
    model = svm.SVC(C=C, gamma=gamma,kernel='rbf',probability=True)
    traindata = data[0:N]
    trainlabel = copy.deepcopy(label[0:N])
    trainlabel[trainlabel != i] = -1
    trainlabel[trainlabel == i] = 1
    trainlabel[trainlabel == -1] = 0
    model.fit(traindata, trainlabel)
    return model

def predict(models,X):
    val = []
    for i in range(10):
        val.append(models[i].predict_proba(X)[:,1])
    val = np.array(val)
    pred = np.argmax(val,axis=0)
    return pred

def sc(y,y_pred):
    l = y.__len__()
    tp = np.count_nonzero(y==y_pred)
    return tp / l

def CrossValidation(data,label,param_grid):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size=0.4, random_state=0)
    C = param_grid['C']
    gamma = param_grid['gamma']
    best_score= 0.0
    best_param = {'C':1,'gamma':0.001}
    for c in tqdm(C):
        for g in gamma:
            models = []
            for i in range(10):
                mod = fit(i,X_train,y_train,c,g)
                models.append(mod)
            y_pred = predict(models,X_test[0:100])
            score = sc(y_test[0:100],y_pred)
            print("c=%f, gamma=%f, score=%f"%(c,g,score))
            if score >= best_score:
                best_score = score
                best_param['C'], best_param['gamma'] = c, g
    return best_param

if __name__ == "__main__":
    data, label = readdata()    # read in order this time
    data[data <= 127] = 0
    data[data > 127] = 1

    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    best_parameters = CrossValidation(data,label,param_grid)
    for c, g in list(best_parameters.items()):
        print(c, g)
    best_parameters = {'C':1000,'gamma':0.001}

    data, label = shuffledata(data,label)
    # traindata, trainlabel = data, label
    traindata, trainlabel = data[0:1000], label[0:1000]
    t_1 = time.clock()
    models = []
    for i in range(10):
        mod = fit(i, traindata, trainlabel, best_parameters['C'], best_parameters['gamma'])
        models.append(mod)
    t_2 = time.clock()
    dt = t_2 - t_1

    t_1 = time.clock()
    test_data, test_label = data[-100:], label[-100:]
    pred_label = predict(models,test_data)
    accuracy = sc(test_label,pred_label)
    t_2 = time.clock()
    dt_test = t_2 - t_1

    print("accuracy = %f, "%(accuracy))
    print("train time = %.4fsec, test time = %.4fsec"%(dt,dt_test))