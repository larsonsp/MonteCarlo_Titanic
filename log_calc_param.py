import scipy.stats as stats
import numpy as np

#CACLULCUALTES THE INPUT FOR LOGNORM

# Desired mean and standard deviation of the log-normal distribution
desired_mean = 29.82596385
desired_std = 38.06890240954017

# Solve for the parameters of the underlying normal distribution
sigma = np.sqrt(np.log((desired_std / desired_mean) ** 2 + 1))
mu = np.log(desired_mean) - sigma ** 2 / 2

# These mu and sigma can be used in your CUDA kernel
print("mu:", mu, "sigma:", sigma)
