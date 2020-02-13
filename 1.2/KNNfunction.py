import numpy as np
import random
from tqdm import tqdm

def KNNfit(traindata, trainlabel):
    trainlist = []
    for id,t in enumerate(traindata):
        trainlist.append([t,trainlabel[id]])    # The format and shape of data: [[data data ...], label], [[28*28], 1]
    return trainlist

def shuffledKNNfit(traindata, trainlabel):
    before_shuffle = list(zip(traindata,trainlabel))
    random.shuffle(before_shuffle)
    shuffled_traindata, shuffled_trainlabel = zip(*before_shuffle)
    return KNNfit(shuffled_traindata, shuffled_trainlabel)

def dist(x1, x2):
    vector = (x1 - x2) ** 2
    sum = np.sum(vector)

    return sum

def pred_in_topK(topK):     # Get the most likely one from top-k
    cnt = 0
    pred = -1
    for item in topK:
        cnt_i = topK.count(item)
        if cnt_i > cnt:
            cnt = cnt_i
            pred = item
    return pred

def KNNpredict(model, testdata, k):     # Model's format is the same as above
    n = len(testdata)
    ypred = []
    for id in tqdm(range(n)):
        temp_dist = []
        for item in model:
            dis = dist(testdata[id],item[0])
            temp_dist.append(dis)
        dist_index = list(np.argsort(temp_dist))
        topK_label = []
        for index in dist_index[0:k]:
            topK_label.append(model[index][1])
        ypred.append(pred_in_topK(topK_label))
    return ypred
