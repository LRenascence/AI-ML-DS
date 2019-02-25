import numpy as np
from pylab import *
from scipy import stats
import matplotlib.pyplot as plt

def predict(x, slope, intercept):
    return slope * x + intercept

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 1, 1000)) * 3


# ordinary least square
[slope, intercept, r_value, p_value, std_err] = stats.linregress(pageSpeeds, purchaseAmount)

# plot
fitLine = predict(pageSpeeds, slope, intercept)
plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c = 'r')
plt.show()