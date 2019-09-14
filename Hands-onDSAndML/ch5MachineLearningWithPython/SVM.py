import numpy as np
from pylab import *
from sklearn import svm, datasets

# create fake cluster data
def createClusteredData(N, k):
    np.random.seed(10)
    pointPerCluster = float(N) / k
    X = []
    Y = []
    for i in range(k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0),
                      np.random.normal(ageCentroid, 2.0)])
            Y.append(i)
    return np.array(X), np.array(Y)

# plot function
def plotPredictions(clf, x, y):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10), np.arange(10, 70, 0.5))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize = (8, 6))
    Z = z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.8)
    plt.scatter(x[:, 0], x[:, 1], c = y.astype(np.float))
    plt.show()

(x, y) = createClusteredData(100, 5)
# SVM algorithm
svc = svm.SVC(kernel = 'linear', C = 1.0).fit(x, y)

plotPredictions(svc, x, y)

print(svc.predict([[200000, 40]]))
print(svc.predict([[50000, 65]]))