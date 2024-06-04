import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *

def my_function(a_Lever, b = 1.1, mean = 0.01, stdev = 0.001,samples_MC = 100):
    nvars = len(a_Lever)
    X = np.zeros(nvars)
    average = np.zeros(nvars)
    a = np.array(a_Lever)
    for _ in range (samples_MC):
        X[0] = 0.0
        X[1] = 0.0
        random_input = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        for t in range(2,nvars):
            float(X[t]) == -(a/b)*X[t-2] - (0.25/b)*X[t-1] + (1/b)*random_input 
            average[t] += X[t]/float(samples_MC)
                
    maxAverage = np.max(average)
    maxX = np.max(X)

    return (maxAverage, maxX, average)


model = Model(my_function)

model.parameters = [Parameter("a_Lever"),
                    Parameter("b"),
                    Parameter("mean"),
                    Parameter("stdev")]

# Define the model outputs
model.responses = [Response("maxAverage", Response.MINIMIZE),
                   Response("maxX", Response.MINIMIZE)]

# Some parameters are levers that we control via our policy
model.levers = [RealLever("a_Lever", 0.1, 0.2, length=100)]

model.uncertainties = [UniformUncertainty("b", 1.1, 2.45),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005)]



# Prepare the cache for storing intermediate results
setup_cache(file="example.cache")

# Optimize the model or get cached results if they exist.  Note that the
# call to optimize is wrapped in a lambda function to enable lazy evaluation.
output = cache("output", lambda: optimize(model, "NSGAII", 10000))

# save the Pareto approximate set as a .csv file
output.save('optimization_results.csv')

   
# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

# Use Seaborn settings for pretty plots
sns.set()

# Plot the points in 2D space
scatter2d(model, output)
plt.show()


