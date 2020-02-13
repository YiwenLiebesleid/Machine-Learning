import scipy.io as scio
import numpy as np
import KNNfunction as KF
import pretreat as pt

if __name__ == "__main__":
    mnist = scio.loadmat("C:\\Users\\HP\\Desktop\\WPI\\19fall\\machine learning\\TeamAssignment\\mnistAll.mat")
    data = mnist['mnist'][0][0]
    traindata = data[0].transpose(2,0,1)
    # traindata = data[0].transpose(2,0,1)[0:6000]
    testdata = data[1].transpose(2,0,1)[0:1000]
    trainlabel = np.hstack(data[2].transpose())
    # trainlabel = np.hstack(data[2].transpose())[0:6000]
    testlabel = np.hstack(data[3].transpose())[0:1000]

    # traindata, testdata = pt.pretreat(traindata,testdata)
    traindata = pt.divider(9,traindata)
    testdata = pt.divider(9,testdata)

    """  # Get the best way to divide
    rawtrain = traindata
    rawtest = testdata
    err = []
    for times in range(2,15):
        traindata = pt.divider(times,rawtrain)
        testdata = pt.divider(times,rawtest)

        k = 3
        model = KF.KNNfit(traindata, trainlabel)
        # model = KF.shuffledKNNfit(traindata,trainlabel)
        ypred = KF.KNNpredict(model, testdata, k)
        diff = list(ypred - testlabel)
        errors = len(diff) - diff.count(0)
        err.append(errors / 100)

    for i in range(len(err)):
        print("%d: err=%.4f"%(i+2,err[i]))
    """

    k = 9

    """  # Get the best k value
    err = []
    for k in range(1,16):
        model = KF.KNNfit(traindata, trainlabel)
        # model = KF.shuffledKNNfit(traindata,trainlabel)
        ypred = KF.KNNpredict(model, testdata, k)
        diff = list(ypred - testlabel)
        errors = len(diff) - diff.count(0)
        err.append(errors / 100)
    for i in range(len(err)):
        print("%d: err=%.4f" % (i + 1, err[i]))

    """
    model = KF.KNNfit(traindata,trainlabel)
    # model = KF.shuffledKNNfit(traindata,trainlabel)
    ypred = KF.KNNpredict(model,testdata,k)
    diff = list(ypred - testlabel)
    errors = len(diff) - diff.count(0)
    err = errors / 1000

    print("%.4f"%(err))
