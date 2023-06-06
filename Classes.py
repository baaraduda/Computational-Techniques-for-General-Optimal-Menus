

import numpy as np
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
from Settingfunctions import *
import matplotlib.pyplot as plt

class GammaDistributions:

    #A collection of continious random objects of scipy.stats with support [a,b]
    def __init__(self, a, b, N):
        self.a = a
        self.b = b
        self.N = N
        #used sigma and varianve for the distiubtions based on a and b
        self.sigma = np.log(b/a)
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

    def EnormCvar(self, sigma):
        return ECDF(stats.truncnorm((self.a - self.mean) / sigma, (self.b - self.mean) / sigma, self.mean, sigma).rvs(int(self.N)))

    def EnormDvar(self, sigma):
        return ECDF(np.concatenate((stats.truncnorm((self.a - self.a) / sigma, (self.b - self.a) / sigma, self.a, sigma).rvs(int(self.N/2))),stats.truncnorm((self.a - self.b) / sigma, (self.b - self.b) / sigma, self.b, sigma).rvs(int(self.N/2))))
    
    def normLcamel(self, sigma):
            mean = self.a + 2 * (self.b-self.a)/5
            return stats.truncnorm((self.a - mean) / sigma, (self.b - mean) / sigma, mean, sigma)

    def normRcamel(self, sigma):
        mean = self.a + 3 * (self.b-self.a)/5
        return stats.truncnorm((self.a - mean) / sigma, (self.b - mean) / sigma, mean, sigma)

    def EnormCamel(self, sigma):
        return ECDF(np.concatenate((self.normLcamel(sigma).rvs(int(self.N/2)), self.normRcamel(sigma).rvs(int(self.N/2)))))


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

#np.random.seed(123)
Gammas = GammaDistributions(1,10,10000)

def GamLabCol(mode):
    """
    mode 1: only uniform
    mode 2: normal center and normal dip
    mode 3: normal left and normal right
    """

    if mode == 1:
        return [Gammas.Eunif()], ['Uniform'] , ['blue']
    if mode == 2:
        return [Gammas.Eunif(), Gammas.EnormC()], ['Uniform', 'Normal Center'] , ['blue', 'red']
    if mode == 3:
        return [Gammas.EnormL(), Gammas.EnormD()], ['Normal Left', 'Normal Right'], ['green', 'blue']
    if mode == 4:
        return [Gammas.EnormCvar(1), Gammas.EnormCvar(2)], ['Normal low variance', 'Normal high variance'], ['green', 'blue']
    if mode == 5:
        return [Gammas.EnormCvar(1), Gammas.EnormCvar(2), Gammas.Eunif()], ['Normal low variance', 'Normal high variance', 'Uniform'], ['green', 'blue', 'red']
    if mode == 6:
        GammaCamel = Gammas.EnormCamel(.5)
        sigma = np.std(GammaCamel.x[1:])
        return [Gammas.EnormCvar(sigma), GammaCamel], ['Normal', 'Camel'], ['green', 'blue']
    if mode == 7:
        normsigma = Gammas.normC().std()
        return [Gammas.EnormC(), Gammas.normC()], ['Emperical Normal', 'Normal'], ['green', 'blue']
    
def showDist(mode):

    GammaSet, Labels, Colors = GamLabCol(mode)
    xx = np.linspace(1,10,51)
    for i in range(len(GammaSet)):
        if isinstance(GammaSet[i], ECDF):
            plt.hist(GammaSet[i].x[1:], bins= 51, label = Labels[i], color = Colors[i], alpha = .5)
            print(f'{Labels[i]}: mean={np.mean(GammaSet[i].x[1:])}, std = {np.std(GammaSet[i].x[1:])}')
        else:
            plt.plot(xx, GammaSet[i].pdf(xx), label = Labels[i],color= Colors[i])
            print(f'{Labels[i]}: mean={GammaSet[i].mean()}, std = {GammaSet[i].std()}')

    plt.show()

n=5
gamma=Gammas.EnormC()         
eta = 1
s1 = S(n= n, Gamma = gamma, eta= 1)


def my_formula( i):
    return  (1**(1-i/n))*(10**(i/n))
benchmark = np.fromfunction(my_formula,(n+1,))

#showDist(7)

