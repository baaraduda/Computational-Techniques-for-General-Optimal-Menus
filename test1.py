from scipy import stats, integrate
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import functions
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import root
import scipy.stats as stats
from tqdm import tqdm
from scipy.stats import truncnorm
from scipy.stats import rv_continuous
import statsmodels.api as sm
import matplotlib.pyplot as plt

#Here are the global parameters:
r = .02 #risk-free rate
mu = .05 #expected return asset
sigma = math.sqrt(.03) #volatility asset
T = 20 #time 
a=1 #lowest possible risk aversion
b=10 #highest possible risk aversion
eta = 2



def optimaldecision(a,b, Gamma):

    def f(m, Gamma):
        def e(g):
            return np.exp(g*.5*(sigma**2)*(eta-1)*T*(m**2))
        if Gamma[-1]==0:
            if len(Gamma)==2:
                integrand1 = lambda g: g * e(g) * Gamma[0].pdf(g)
                E1,_ = integrate.quad(integrand1, a, b)
                integrand2 = lambda g: e(g) * Gamma[0].pdf(g)
                E2,_ = integrate.quad(integrand2, a, b)

                return m-(mu-r)/((sigma**2)*E1/E2)
            
            if len(Gamma)==3:
                integrand1 = lambda g: g * e(g) * .5 * (Gamma[0].pdf(g) + Gamma[1].pdf(g))
                E1,_ = integrate.quad(integrand1, a, b)
                integrand2 = lambda g: e(g) * .5 * (Gamma[0].pdf(g) + Gamma[1].pdf(g))
                E2,_ = integrate.quad(integrand2, a, b)

                return m-(mu-r)/((sigma**2)*E1/E2)
        else:
            indices = np.where((Gamma[0].x >= a) & (Gamma[0].x <= b))[0]
            E1 = np.sum(Gamma[0].x[indices] * e(Gamma[0].x[indices])) / len(Gamma[0].x[indices])
            E2 = np.sum(e(Gamma[0].x[indices])) / len(Gamma[0].x[indices])

            return m-(mu-r)/((sigma**2)*E1/E2)

    x0 = .5 * ((mu-r)/(b*(sigma**2) + (mu-r)/(a*(sigma**2))))

    sol = root(f, x0)

    return sol.x[0]



def f(m, Gamma):
    def e(g):
        return np.exp(g*.5*(sigma**2)*(eta-1)*T*(m**2))
    if Gamma[-1]==0:
        if len(Gamma)==2:
            integrand1 = lambda g: g * e(g) * Gamma[0].pdf(g)
            E1,_ = integrate.quad(integrand1, a, b)
            integrand2 = lambda g: e(g) * Gamma[0].pdf(g)
            E2,_ = integrate.quad(integrand2, a, b)

            return m-(mu-r)/((sigma**2)*E1/E2)
        
        if len(Gamma)==3:
            integrand1 = lambda g: g * e(g) * .5 * (Gamma[0].pdf(g) + Gamma[1].pdf(g))
            E1,_ = integrate.quad(integrand1, a, b)
            integrand2 = lambda g: e(g) * .5 * (Gamma[0].pdf(g) + Gamma[1].pdf(g))
            E2,_ = integrate.quad(integrand2, a, b)

            return m-(mu-r)/((sigma**2)*E1/E2)
    else:
        indices = np.where((Gamma[0].x >= a) & (Gamma[0].x <= b))[0]
        E1 = np.sum(Gamma[0].x[indices] * e(Gamma[0].x[indices])) / len(Gamma[0].x[indices])
        E2 = np.sum(e(Gamma[0].x[indices])) / len(Gamma[0].x[indices])

        return m-(mu-r)/((sigma**2)*E1/E2)

Unif = stats.uniform(a,b-a)

Gamma0 = [Unif,0]

data = Unif.rvs(10000)
Eunif = ECDF(data)

Gamma1 = [Eunif,1]

print(f(.5,Gamma0))
print(f(.5,Gamma1))