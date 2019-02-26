import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds
# split the data into train and test
# 80% for train, 20% for test
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]
trainY = purchaseAmount[: 80]
testY = purchaseAmount[80:]

x = np.array(trainX)
y = np.array(trainY)
# use polynomial regression
p4 = np.poly1d(np.polyfit(x, y, 6))
# plot
xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(testX, testY)
plt.plot(xp, p4(xp), c = 'r')
plt.show()
# get the r-squared
r2 = r2_score(testY, p4(testX))
print(r2)