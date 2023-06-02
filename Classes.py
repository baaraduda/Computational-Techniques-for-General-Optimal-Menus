

import numpy as np
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
from Settingfunctions import *

class GammaDistributions:

    #A collection of continious random objects of scipy.stats with support [a,b]
    def __init__(self, a, b, N):
        self.a = a
        self.b = b
        self.N = N
        #used sigma and varianve for the distiubtions based on a and b
        self.sigma = np.log(b/(a))
        self.mean = (a+b)/2

    def unif(self):
        return stats.uniform(self.a,self.b-self.a)
    
    def Eunif(self):
        return ECDF(self.unif().rvs(self.N))
    
    def normC(self):
        return stats.truncnorm((self.a - self.mean) / self.sigma, (self.b - self.mean) / self.sigma, self.mean, self.sigma)

    def EnormC(self):
        return ECDF(self.normC().rvs(self.N))

    def normL(self):
        return stats.truncnorm((self.a - self.a) / self.sigma, (self.b - self.a) / self.sigma, self.a, self.sigma)

    def EnormL(self):
        return ECDF(self.normL().rvs(self.N))

    def normR(self):
        return stats.truncnorm((self.a - self.b) / self.sigma, (self.b - self.b) / self.sigma, self.b, self.sigma)

    def EnormR(self):
        return ECDF(self.normR().rvs(self.N))
    

    def EnormD(self):
        return ECDF(np.concatenate((self.normL().rvs(int(self.N/2)), self.normR().rvs(int(self.N/2)))))




class S:
    def __init__(self, n, Gamma, eta, T = 20, r = 0.02, mu = 0.05, sigma = np.sqrt(.03)):
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
