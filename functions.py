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
from statsmodels.distributions.empirical_distribution import ECDF

##All functions used in the computations and all global parameters

#Here are the global parameters:
r = .02 #risk-free rate
mu = .05 #expected return asset
sigma = math.sqrt(.03) #volatility asset
T = 20 #time 
a=1 #lowest possible risk aversion
b=10 #highest possible risk aversion


##And the important ones:

#Risk aversion for the analysis of a single investor
gamma = 3 

#Here follow different possible gamma distributions on the [a,b] support with the relevant parameters.
sig= math.log(b/a) #own choice how sigma relates to a and b
mean = (a+b)/2 #own choice how the mean relates to a and b


#risk aversion of the social planner
eta = 1

#the amount of optimal decisions the social planner can make for his population
n=5

#the benchmark is the optimal partition for eta=1 and gamma uniform
def my_formula(i):
    return  (a**(1-i/n))*(b**(i/n))
benchmark = np.fromfunction(my_formula,(n+1,))

#guessP is the initial guess for the gradient descent algorithm
guessP= np.linspace(a,b,n+1) 

class MixtureDistribution(rv_continuous):
    def __init__(self, dist1, dist2, w1):
        self.dist1 = dist1
        self.dist2 = dist2
        self.w1 = w1
        super(MixtureDistribution, self).__init__()

    def _pdf(self, x):
        return self.w1 * self.dist1.pdf(x) + (1 - self.w1) * self.dist2.pdf(x)

def invsp(v, eta):
    if eta == 1:
        return math.e**v
    else:
        return ((1-eta)*v+1)**(1/(1-eta))

def optimaldecision(a,b, Gamma):

    def f(m):
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

def goal_function(pg,eta,Gamma):
    sum = 0

    def Usp(eta,c):
        """returns utility social planner for given eta and certainty equivalent c""" 
        if eta == 1:
            return np.log(c)
        else:
            return (c**(1-eta)-1)/(1-eta)

    for i in range(1, len(pg)):
        mi = optimaldecision(pg[i-1],pg[i], Gamma)
        def vce(g):
            return Usp(eta, np.exp(r*T + (mu - r)*mi*T-.5*g*(sigma**2)*(mi**2)*T))   
        if Gamma[-1]==0:     
            integrand = lambda g: vce(g)*Gamma[0].pdf(g)
            E,_ = integrate.quad(integrand, pg[i-1], pg[i])
            sum += E
        else:
            indices = np.where((Gamma[0].x >= pg[i-1]) & (Gamma[0].x <= pg[i]))[0]
            E = np.sum(vce(Gamma[0].x[indices])) / len(Gamma[0].x)
            sum +=E
    return sum

def f(pg, i, Gamma):
    g = lambda m :(mu-r)/(m*(sigma**2))
    H = lambda x,y: 2/((1/x)+(1/y))
    if i == 0 or i == len(pg)-1:
        return 0
    gmi=g(optimaldecision(pg[i-1], pg[i], Gamma))
    gmi_1=g(optimaldecision(pg[i], pg[i+1], Gamma))

    return  (H(gmi,gmi_1)-pg[i])**2

def objective_function(pg, Gamma):

    f_values = [f(pg,i,Gamma) for i in range(1, len(pg)-1)]

    return sum(f_values)

def parttodecision(partition):
    return [optimaldecision(partition[i],partition[i+1]) for i in range(len(partition))]

def AdamAlgorithm(a,b, eta, Gamma, guess=np.linspace(a,b,n+1), learning_rate=0.1, max_iterations=1000, epsilon=1e-8, tolerance=1e-3, beta1=0.9, beta2=0.999):

    def optimaldecision(a,b):

        def f(m):
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

    def f(pg, i):
        g = lambda m :(mu-r)/(m*(sigma**2))
        H = lambda x,y: 2/((1/x)+(1/y))
        if i == 0 or i == len(pg)-1:
            return 0
        gmi=g(optimaldecision(pg[i-1], pg[i]))
        gmi_1=g(optimaldecision(pg[i], pg[i+1]))

        return  (H(gmi,gmi_1)-pg[i])**2

    def objective_function(pg):

        f_values = [f(pg,i) for i in range(1, len(pg)-1)]

        return sum(f_values)
    
    def gradient(f, x):
        """

        """
        eps = 1e-6  # small value for epsilon
        n = x.shape[0]  # number of dimensions
        
        grad = np.zeros(n)
        for i in range(n):
            x_eps = x.copy()
            x_eps[i] += eps
            grad[i] = (f(x_eps) - f(x)) / eps
            
        return grad
    """
    Adam gradient descent algorithm for optimizing the objective function
    """
    # Set the initial guess for the partition
    partition = guess
    guesspartition = guess.copy()
    # Initialize the moment estimates for the gradient and its squared magnitude
    m = np.zeros_like(partition)
    v = np.zeros_like(partition)

    path = []
    path.append(partition.copy()) # add the initial guess to the path
    
    maxgradold = 9999

    for i in tqdm(range(max_iterations)):
        # Compute the gradient of the objective function
        grad = gradient(objective_function, partition)
        maxgradnew = np.abs(grad).max()

        d =1
        if maxgradnew > 2 * maxgradold:
            print("   doop")
            d = maxgradold/maxgradnew
        
        # Update the moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias-correct the moment estimates
        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1))
        
        # Adjust the partition using the Adam update rule
        partition[1:-1] -= d* learning_rate * m_hat[1:-1] / (np.sqrt(v_hat[1:-1]) + epsilon)

        # Enforce the constraints a=g0 and gn=b
        partition[0] = a
        partition[-1] = b

        if i > 3:
            difference = np.sum(np.abs(np.array(partition)-np.array(path[-2])))
            if difference <  tolerance:
                print('ploop')
                partition = .5 * (np.array(partition)+np.array(path[-1]))


        path.append(partition.copy())

        # Check whether partition is in ascending order
        if np.all(np.diff(path) > 0) == False:
            print('unfeasible region')
            break

        # Check for convergence
        if np.abs(grad).max() < tolerance:
            path.append(partition.copy())
            break
            
        if i == max_iterations-1:
            print(path[-1])
            break
            #return GDAlgorithm(a,b,eta, Gamma, guesspartition)

        maxgradold = maxgradnew
    return path

def GDAlgorithm(a,b,eta, Gamma, guess=np.linspace(a,b,n+1), learning_rate=1, max_iterations=1000,tolerance=1e-6):
    def optimaldecision(a,b):

        def f(m):
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

    def f(pg, i):
        g = lambda m :(mu-r)/(m*(sigma**2))
        H = lambda x,y: 2/((1/x)+(1/y))
        if i == 0 or i == len(pg)-1:
            return 0
        gmi=g(optimaldecision(pg[i-1], pg[i]))
        gmi_1=g(optimaldecision(pg[i], pg[i+1]))

        return  (H(gmi,gmi_1)-pg[i])**2

    def objective_function(pg):

        f_values = [f(pg,i) for i in range(1, len(pg)-1)]

        return sum(f_values)
    
    def gradient(f, x):
        """

        """
        eps = 1e-6  # small value for epsilon
        n = x.shape[0]  # number of dimensions
        
        grad = np.zeros(n)
        for i in range(n):
            x_eps = x.copy()
            x_eps[i] += eps
            grad[i] = (f(x_eps) - f(x)) / eps
            
        return grad
    
    # Set the initial guess for the partition
    partition = guess

    path = []
    path.append(partition.copy()) # add the initial guess to the path
    maxgradold = 9999

    for i in tqdm(range(max_iterations)):
        # Compute the gradient of the objective function
        grad = gradient(objective_function, partition)
        maxgradnew = np.abs(grad).max() 

        d =1
        if maxgradnew > 2 * maxgradold:
            print("   doop")
            d = maxgradold/maxgradnew
        # Adjust the partition using the gradient
        partition[1:-1] -= learning_rate * grad[1:-1] * d

        # Enforce the constraints a=g0 and gn=b
        partition[0] = a
        partition[-1] = b


        if i > 3:
            difference = np.sum(np.abs(np.array(partition)-np.array(path[-2])))
            if difference <  1e-3 *tolerance:
                print('   ploop')
                partition = .5 * (np.array(partition)+np.array(path[-1]))

        path.append(partition.copy())

        #Check whether partition is in ascending order
        if np.all(np.diff(path) > 0) == False:
            print('  infeasible region')
            partition = .5 * (np.array(path[-2])+np.array(path[-1]))


        # Check for convergence
        if maxgradnew < tolerance:
            path.append(partition.copy())
            break
            
        if i == max_iterations:
            path.append(partition.copy())

        maxgradold = maxgradnew
    return path

d