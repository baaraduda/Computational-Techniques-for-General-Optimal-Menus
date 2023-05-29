import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
from tqdm import tqdm
from scipy.stats import truncnorm
from scipy.stats import rv_continuous
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


class GammaDistributions:

    #A collection of continious random objects of scipy.stats with support [a,b]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        #used sigma and varianve for the distiubtions based on a and b
        self.sigma = math.log(b/a)
        self.mean = (a+b)/2

    def unif(self):
        return stats.uniform(self.a,self.b-self.a)
    
    def Eunif(self, N=1000):
        return ECDF(self.unif().rvs(N))
    
    def normC(self):
        return stats.truncnorm((self.a - self.mean) / self.sigma, (self.b - self.mean) / self.sigma, self.mean, self.sigma)

    def EnormC(self, N=1000):
        return ECDF(self.normC().rvs(N))

    def normL(self):
        return stats.truncnorm((self.a - self.a) / self.sigma, (self.b - self.a) / self.sigma, self.a, self.sigma)

    def EnormL(self, N=1000):
        return ECDF(self.normL().rvs(N))

    def normR(self):
        return stats.truncnorm((self.a - self.b) / self.sigma, (self.b - self.b) / self.sigma, self.b, self.sigma)

    def EnormR(self, N=1000):
        return ECDF(self.normR().rvs(N)) 
class S:
    def __init__(self, n, Gamma, eta, T = 20, r = 0.02, mu = 0.05, sigma = math.sqrt(.03)):
        self.n = n
        self.Gamma = Gamma
        self.eta = eta
        self.T = T
        self.r = r
        self.mu = mu
        self.sigma = sigma

        if isinstance(self.Gamma, ECDF):
            self.a, self.b = np.min(Gamma.x[1:-1]), np.max(Gamma.x[1:-1])
        else:
            self.a, self.b = Gamma.ppf(0), Gamma.ppf(1)

    def mOpt(self, g):
        return (self.mu - self.r)/((self.sigma**2)*g)
    def gOpt(self, m):
        return (self.mu - self.r)/((self.sigma**2)*m)

    def optimalDecision(self, l=None, u=None):
        if l is None:
            l = self.a
        if u is None:
            u = self.b

        x0 = 0.5 * (self.mOpt(l) + self.mOpt(u))
        interval = [self.mOpt(l), self.mOpt(u)]

        if isinstance(self.Gamma, ECDF):
            indices = np.where((self.Gamma.x >= l) & (self.Gamma.x <= u))[0]
            gamma_x = self.Gamma.x[indices]
            gamma_len = len(gamma_x)
            e_values = np.exp(gamma_x * 0.5 * (self.sigma ** 2) * (self.eta - 1) * self.T * x0 ** 2)
            E1 = np.sum(gamma_x * e_values) / gamma_len
            E2 = np.sum(e_values) / gamma_len

            def f(m):
                return m - (self.mu - self.r) / ((self.sigma ** 2) * E1 / E2)

        else:
            integrand1 = lambda g: g * np.exp(g * 0.5 * (self.sigma ** 2) * (self.eta - 1) * self.T * x0 ** 2) * self.Gamma.pdf(g)
            E1 = integrate.romberg(integrand1, l, u)

            integrand2 = lambda g: np.exp(g * 0.5 * (self.sigma ** 2) * (self.eta - 1) * self.T * x0 ** 2) * self.Gamma.pdf(g)
            E2 = integrate.romberg(integrand2, l, u)

            def f(m):
                return m - (self.mu - self.r) / ((self.sigma ** 2) * E1 / E2)

        result = optimize.root_scalar(f, method='brentq', bracket=interval, x0=x0)
        return result.root

    def systemM1(self, m):
        residuals = np.zeros(self.n-1)

        for i in range(1, self.n):
            m1 = self.optimalDecision(self.gOpt(m[i-1]), self.gOpt(m[i]))
            m2 = self.optimalDecision(self.gOpt(m[i]), self.gOpt(m[i+1]))
            residuals[i-1] = m[i] - 0.5 * (m1 + m2)

        return residuals
           
    def optM(self):
        def fun(m):
            return np.sum((self.systemM1(m))**2)
        guess = np.linspace(self.mOpt(self.b),self.mOpt(self.a), self.n+1)
        sol = optimize.minimize(fun=fun,x0=guess)


        return sol.x
    def Adam(self, tolerance = 1e-3, max_iterations=1000,learning_rate=0.1, epsilon=1e-8, beta1=0.9, beta2=0.999):
        initialGuess=np.linspace(self.a,self.b,self.n+1)



Gammas = GammaDistributions(1,10) #we work with support [a=1, b=10] for gamma
N = 1000
s1 = S(n= 3, Gamma = Gammas.unif(), eta= 1)

print(s1.optM()) 
