
def pretreat(traindata, testdata):
    sum = 0
    sum += traindata.sum()
    sum += testdata.sum()
    average = sum / ((len(traindata) + len(testdata)) * 28 * 28)
    for i in range(len(traindata)):
        s = traindata[i].sum()
        ave = s / (28 * 28)
        traindata[i] = traindata[i] * average / ave

    for i in range(len(testdata)):
        s = testdata[i].sum()
        ave = s / (28 * 28)
        testdata[i] = testdata[i] * average / ave

    return traindata, testdata

def divider(n, data):
    data = data // (256 // n + 1)
    return data
