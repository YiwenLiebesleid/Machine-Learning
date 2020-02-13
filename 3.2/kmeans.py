import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
import scipy.io as scio
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def shuffledata(traindata,trainlabel):
    before_shuffle = list(zip(traindata, trainlabel))
    random.shuffle(before_shuffle)
    shuffle_data, shuffle_label = zip(*before_shuffle)
    return shuffle_data, shuffle_label

def getsubset(predict,label,k):
    ret = []
    for i in range(0,k):
        ret.append([])
    for id,j in enumerate(predict):
        ret[j].append(label[id])
    return ret

def purity(subsets,k,N):
    sum = 0
    for i in range(0,k):
        subi_cnt = subsets[i].count(max(subsets[i],key=subsets[i].count))   # get the number of the element occurs the most times in subset i.
        sum += subi_cnt
    return sum / N

def readdata():
    mnist = scio.loadmat("C:\\Users\\HP\\Desktop\\WPI\\19fall\\machine learning\\TeamAssignment\\2\\mnist-original.mat")
    data = mnist['data'].transpose()
    label = mnist['label'].transpose().ravel()  # not used in this assignment
    data, label = shuffledata(data, label)
    return np.array(data), np.array(label)

if __name__=="__main__":
    data, label = readdata()
    subdata = data[0:3000]
    sublabel = label[0:3000]

    # data = data[0:5000]
    # label = label[0:5000]
    label = np.array(label).astype(int)

    result_list = []

    for k in tqdm(range(2,11)):
        k_model = KMeans(n_clusters=k,init='k-means++',random_state=0).fit(subdata)
        s = metrics.calinski_harabaz_score(subdata,k_model.labels_)
        result_list.append(s)

    best_k = result_list.index(max(result_list)) + 2

    t_1 = time.clock()
    kmeans_model = KMeans(n_clusters=best_k, init='k-means++', random_state=0).fit(data)
    predict = kmeans_model.labels_
    t_2 = time.clock()
    dt = t_2 - t_1
    subsets = getsubset(kmeans_model.labels_,label,best_k)
    prt = purity(subsets,best_k,len(label))
    s = metrics.calinski_harabaz_score(data,kmeans_model.labels_)
    print("Purity of k = %d is %lf, Calinski-Harabaz score is %lf, %lfsec used"%(best_k,prt,s,dt))


    color_set = [[0.2157,0.4941,0.7216],[0.3020,0.6863,0.2902],
                 [0.5961,0.3059,0.6392],[1,0.4980,0],[1,1,0.2],
                 [0.6510,0.3373,0.1569],[0.9686,0.5059,0.7490],
                 [0.6,0.6,0.6],[0.5,0.6,0.3],[0.6,0.2,0.4]]

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(10,10))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i,0], X_norm[i,1], str(label[i]),color=color_set[predict[i]],fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()