
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from functions import *


class GammaDistributions:

    #A collection of continious random objects of scipy.stats with support [a,b]
    def __init__(self, a, b):
        self.a = a
        self.b = b
        #used sigma and varianve for the distiubtions based on a and b
        self.sigma = np.log(b/a)
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
    def __init__(self, n, Gamma, eta, T = 20, r = 0.02, mu = 0.05, sigma = np.sqrt(.03)):
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
            E1, _ = integrate.quad(integrand1, l, u)

            integrand2 = lambda g: np.exp(g * 0.5 * (self.sigma ** 2) * (self.eta - 1) * self.T * x0 ** 2) * self.Gamma.pdf(g)
            E2, _ = integrate.quad(integrand2, l, u)

            def f(m):
                return m - (self.mu - self.r) / ((self.sigma ** 2) * E1 / E2)

        result = optimize.root_scalar(f, method='brentq', bracket=interval, x0=x0)
        return result.root
     
    def objM(self, m):
        residuals = np.zeros(self.n-1)
        for i in range(1, self.n):
            m1 = self.optimalDecision(self.gOpt(m[i-1]), self.gOpt(m[i]))
            m2 = self.optimalDecision(self.gOpt(m[i]), self.gOpt(m[i+1]))
            residuals[i-1] = m[i] - 0.5 * (m1 + m2)
        return np.sum(residuals**2)
    def objG(self, g):
        H = lambda x,y: 2/((1/x)+(1/y))
        residuals = np.zeros(self.n-1)
        for i in range(1,self.n):
            g1 = self.gOpt(self.optimalDecision(g[i-1],g[i]))
            g2 = self.gOpt(self.optimalDecision(g[i],g[i+1]))
            residuals[i-1] = (g[i]-H(g1,g2))**2
        check = np.sum(residuals)
        return np.sum(residuals)
    def gradient(self, f, x):
        """

        """
        eps = 1e-6  # small value for epsilon
        n = x.shape[0]  # number of dimensions
        
        grad = np.zeros(n)
        for i in range(n):
            x_eps = x.copy()
            x_eps[i] += eps
            f1 = f(x)
            f2= f(x_eps)
            s1 = self.objG(x)
            s2 = self.objG(x_eps)
            grad[i] = (f(x_eps) - f(x)) / eps
            
        return grad
    
    def Adam(self, allpath = 0, tolerance = 1e-3, max_iterationlon=100, beta1=0.9, beta2=0.999):
        initialGuess=np.linspace(self.a,self.b,self.n+1)
        return 
    
    def GD(self, allpath = 0, tolerance = 1e-3, max_iterations = 100, learning_rate=1):
        # Set the initial guess for the partition
        partition = np.linspace(self.a, self.b, self.n+1)
        path = []
        path.append(partition.copy()) # add the initial guess to the path
        maxgradold = 9999
        def f(pg, i):
            g = lambda m :(mu-r)/(m*(sigma**2))
            H = lambda x,y: 2/((1/x)+(1/y))
            if i == 0 or i == len(pg)-1:
                return 0
            gmi=g(optimaldecision(pg[i-1], pg[i],[self.Gamma]))
            gmi_1=g(optimaldecision(pg[i], pg[i+1],[self.Gamma]))

            return  (H(gmi,gmi_1)-pg[i])**2

        def objective_function(pg):

            f_values = [f(pg,i) for i in range(1, len(pg)-1)]

            return sum(f_values)
        
        for i in tqdm(range(max_iterations)):
            # Compute the gradient of the objective function
            grad = self.gradient(objective_function, partition)
            maxgradnew = np.abs(grad).max() 

            d =1
            if maxgradnew > 2 * maxgradold:
                print("   doop: took a little")
                d = maxgradold/maxgradnew
            # Adjust the partition using the gradient
            partition[1:-1] -= learning_rate * grad[1:-1] * d

            # Enforce the constraints a=g0 and gn=b
            partition[0] = self.a
            partition[-1] = self.b


            if i > 3:
                difference = np.sum(np.abs(np.array(partition)-np.array(path[-2])))
                if difference <  1e-6 *tolerance:
                    print('   ploop: took the middle')
                    frac = np.random.rand(1) 
                    perb = 1e-1
                    partition = frac * np.array(path[-2])+ (1 - frac) * np.array(path[-1])
                    + np.random.uniform(-perb,perb,size=n+1)
            path.append(partition.copy())

            #Check whether partition is in ascending order
            if np.all(np.diff(path) > 0) == False:
                print('  infeasible region')
                frac = np.random.rand(1) 
                perb = 1e-1
                partition = frac * np.array(path[-2])+ (1 - frac) * np.array(path[-1])
                + np.random.uniform(-perb,perb,size=n+1)

            # Check for convergence
            if maxgradnew < tolerance:

                path.append(partition.copy())
                break
                
            if i == max_iterations:
                path.append(partition.copy())
            maxgradold = maxgradnew

        if allpath == 0:
            return path[-1]
        else:
            return path



#s2 = S(n= 3, Gamma = Gammas.Eunif(), eta= 1)
#print(s2.GD()) 











"""
path = np.empty((max_iterations, self.n+1))
        path[0] = g.copy() # add the initial guess to the path
        maxgradold = 9999 #always accept the first computed gradient

        for i in tqdm(range(max_iterations)):
            # Compute the gradient of the objective function
            gradient = grad(self.objG,g)

            maxgradnew = np.abs(gradient).max() 

            d =1
            if maxgradnew > 2 * maxgradold:
                print("   doop: took a little")
                d = maxgradold/maxgradnew
            # Adjust the partition using the gradient
            g[1:-1] -= learning_rate * gradient[1:-1] * d

            # Enforce the constraints a=g0 and gn=b
            g[0] = self.a
            g[-1] = self.b


            if i > 3:
                difference = np.sum(np.abs(np.array(g)-np.array(path[-2])))
                if difference <  1e-3 *tolerance:
                    print('   ploop: took the middle')
                    g = .5 * (np.array(g)+np.array(path[-1]))

            #Check whether partition is in ascending order
            if np.all(np.diff(path) > 0) == False:
                print('  infeasible region')
                g = .5 * (np.array(path[-2])+np.array(path[-1]))


            # Check for convergence
            if maxgradnew < tolerance:
                path[i]=g.copy()
                break
                
            if i == max_iterations:
                path[i]=g.copy()
                break

            path[i]=g.copy()
            maxgradold = maxgradnew

        return path

"""    
"""
    def optM(self):

        linear_constraint_matrix = np.diag(-1 * np.ones(self.n - 1), k=0) + np.diag(np.ones(self.n - 2), k=1)
        np.fill_diagonal(linear_constraint_matrix, -1)
        np.fill_diagonal(linear_constraint_matrix[:, 1:], 1)
        def constraint_func(m):
            constraint_matrix = np.diag(np.ones(self.n-1), k=1)  # Upper triangular matrix
            constraint_vector = np.zeros(self.n-1)
            return constraint_matrix, constraint_vector
        guess = np.linspace(self.mOpt(self.b),self.mOpt(self.a), self.n+1)
        constraints = optimize.LinearConstraint(*constraint_func(guess))

        sol = optimize.minimize(fun=objM,x0=guess,constraints=constraints)


        return sol.x
    
 """