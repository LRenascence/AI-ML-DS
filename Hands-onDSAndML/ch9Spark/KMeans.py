from pyspark.mllib.clustering import KMeans
from numpy import array, random
from math import sqrt
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import scale

# set the k
K = 5

# Spark object
conf = SparkConf().setMaster('local').setAppName('SparkKMeans')
sc = SparkContext(conf = conf)
# generate fake cluster data
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N) / k
    X = []
    # training
    for i in range(k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0),
                     random.normal(ageCentroid, 2.0)])
    return array(X)

# K-Means algorithm
data = sc.parallelize(scale(createClusteredData(100, K)))

clusters = KMeans.train(data, K, maxIterations = 10, initializationMode = 'random')

resultRDD = data.map(lambda point: clusters.predict(point)).cache()

# write result to the file
outputFile = open('output.txt','wt')
outputFile.write('counts by value:')
counts = resultRDD.countByValue()
outputFile.write(str(counts))

outputFile.write('cluster assignments:')
results = resultRDD.collect()
outputFile.write(str(results))

# get the error ratio
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
outputFile.write(('within set sum of squraed error = ' + str(WSSSE)))
