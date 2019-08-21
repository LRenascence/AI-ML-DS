from numpy import random, array, float
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
# generate fake cluster data
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N) / k
    X = []
    for i in range(k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0),
                     random.normal(ageCentroid, 2.0)])
    return array(X)

# K-Means algorithm
data = createClusteredData(100, 5)
#print(data)
model = KMeans(n_clusters = 5)
# scaling data for better result
model = model.fit(scale(data))
print(model.labels_)
# plot
plt.scatter(data[:,0], data[:,1], c = model.labels_.astype(float))
plt.show()
