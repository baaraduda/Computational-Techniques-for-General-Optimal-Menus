import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import root
import scipy.stats as stats
from tqdm import tqdm
from scipy.stats import truncnorm
from scipy.stats import rv_continuous
import numpy as np
from Settingfunctions import *
from Classes import *
import matplotlib.pyplot as plt
Gammas = GammaDistributions(1,10, 1000) #we work with support [a=1, b=10] for gamma
N = 10000
a=1
b=10
n=2
gammaE=Gammas.unif()
eta = 1
s1 = S(n= n, Gamma = gammaE, eta= 1)
def my_formula(i):
    return  (a**(1-i/n))*(b**(i/n))
benchmark = np.fromfunction(my_formula,(n+1,))

class S:
    def __init__(self, n, Gamma, eta, T=20, r=0.02, mu=0.05, sigma=np.sqrt(.03)):
        self.n = n
        self.Gamma = Gamma
        self.eta = eta
        self.T = T
        self.r = r
        self.mu = mu
        self.sigma = sigma

        if isinstance(self.Gamma, ECDF):
            self.a, self.b = round(np.min(Gamma.x[1:-1])), round(np.max(Gamma.x[1:-1]))
        else:
            self.a, self.b = Gamma.ppf(0), Gamma.ppf(1)

def Usp(S, c):
    """returns utility social planner for given eta and certainty equivalent c"""
    eta = S.eta
    if eta == 1:
        return np.log(c)
    else:
        return (c**(1-eta)-1)/(1-eta)

# Create an instance of the S class
n = 1

# Call the Usp function with the s_obj instance
c = 10  # Specify the value of c
utility = Usp(s1, c)
print(utility)


import numpy as np
from scipy.stats import truncnorm

# Define the interval [a, b]
a = -2
b = 2

# Define the mean and variance
mean = 0
variance = 1

# Create the first truncated distribution
dist1 = truncnorm((a - mean) / np.sqrt(variance),
                  (b - mean) / np.sqrt(variance),
                  loc=mean, scale=np.sqrt(variance))
shape1 = (a - mean) / np.sqrt(variance)
shape2 = (b - mean) / np.sqrt(variance)
# Create the second truncated distribution
# Create the second truncated distribution with different shape parameters
dist2 = truncnorm(2 * shape1, 2 * shape2, loc=mean, scale=np.sqrt(variance))


# Print the PDFs of the two distributions at a few points
x = np.linspace(a, b, 100)
pdf1 = dist1.pdf(x)
pdf2 = dist2.pdf(x)

plt.plot(x, pdf1)
plt.plot(x, pdf2)
plt.show()
