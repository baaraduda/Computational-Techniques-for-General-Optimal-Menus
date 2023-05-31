import numpy as np
import scipy.integrate as integrate
from scipy.optimize import root
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

##Auxiliary functions

def mOpt(S, g):
    return (S.mu - S.r)/((S.sigma**2)*g)
def gOpt(S, m):
    return (S.mu - S.r)/((S.sigma**2)*m)

def Usp(S, c):
    """returns utility social planner for given eta and certainty equivalent c""" 
    eta = S.eta
    if eta == 1:
        return np.log(c)
    else:
        return (c**(1-eta)-1)/(1-eta)
    
def invUsp(S, u):
    eta = S.eta
    if eta == 1:
        return np.e(u)
    else:
        return ((1-eta)*u + 1) ** (1 / 1- eta)

def RiskPartitionToMenu(S,partition):
    return [optimalDecision(S, partition[i],partition[i+1]) for i in range(len(partition))]

def DecisionPartitionToMenu(S, partition):
    return RiskPartitionToMenu(S, gOpt(partition))

def label(x):
    if x == objG:
        return 'in terms of G'
    if x == objM:
        return 'in terms of M'
    if x == Adam:
        return 'Adam'
    if x == GD:
        return 'GD'


def goal_function(S, pg):
    eta = S.eta
    Gamma = S.Gamma
    sigma = S.sigma
    mu = S.mu
    r = S.r
    T = S.T

    sum = 0


    for i in range(1, len(pg)):
        mi = optimalDecision(S, pg[i-1],pg[i])
        def vce(g):
            return Usp(eta, np.exp(r*T + (mu - r)*mi*T-.5*g*(sigma**2)*(mi**2)*T)) 
        if isinstance(Gamma, ECDF):
            indices = np.where((Gamma[0].x >= pg[i-1]) & (Gamma[0].x <= pg[i]))[0]
            E = np.sum(vce(Gamma[0].x[indices])) / len(Gamma[0].x)
            sum +=E          
        if Gamma[-1]==0:     
            integrand = lambda g: vce(g)*Gamma[0].pdf(g)
            E,_ = integrate.quad(integrand, pg[i-1], pg[i])
            sum += E

    return sum


#Algorithm section--
def optimalDecision(S, l=None, u=None):
    if l is None:
        l = S.a
    if u is None:
        u = S.b
    eta = S.eta
    Gamma = S.Gamma
    sigma = S.sigma
    mu = S.mu
    r = S.r
    T = S.T

    def f(m):
        def e(g):
            return np.exp(g*.5*(sigma**2)*(eta-1)*T*(m**2))
        if isinstance(Gamma, ECDF):
            indices = np.where((Gamma.x >= l) & (Gamma.x <= u))[0]
            E1 = np.sum(Gamma.x[indices] * e(Gamma.x[indices])) / len(Gamma.x[indices])
            E2 = np.sum(e(Gamma.x[indices])) / len(Gamma.x[indices])
            return m-(mu-r)/((sigma**2)*E1/E2)        
        
        else:
            integrand1 = lambda g: g * e(g) * Gamma.pdf(g)
            E1,_ = integrate.quad(integrand1, l, u)
            integrand2 = lambda g: e(g) * Gamma.pdf(g)
            E2,_ = integrate.quad(integrand2, l, u)

            return m-(mu-r)/((sigma**2)*E1/E2)


    x0 = .5 * ((mu-r)/(u*(sigma**2) + (mu-r)/(l*(sigma**2))))

    sol = root(f, x0)

    return sol.x[0]

def objG(S, g):
    n = S.n
    H = lambda x,y: 2/((1/x)+(1/y))
    residuals = np.zeros(n-1)
    for i in range(1,n):
        g1 = gOpt(S, optimalDecision(S, g[i-1],g[i]))
        g2 = gOpt(S, optimalDecision(S, g[i],g[i+1]))
        residuals[i-1] = (g[i]-H(g1,g2))**2
    return np.sum(residuals)

def objM(S, m):
    n = S.n
    residuals = np.zeros(n-1)
    for i in range(1, n):
        m1 = optimalDecision(S, gOpt(S, m[i]), gOpt(S, m[i-1]))
        m2 = optimalDecision(S, gOpt(S, m[i+1]), gOpt(S, m[i]))
        residuals[i-1] = m[i] - 0.5 * (m1 + m2)
    return np.sum(residuals**2)

def gradient(S, f, x):
    """

    """
    eps = 1e-6  # small value for epsilon
    n = x.shape[0]  # number of dimensions
    
    grad = np.zeros(n)
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (f(S, x_eps) - f(S, x)) / eps
        
    return grad

#Algorithms:

def GD(S, obj = objG, allpath =0, tolerance=1e-6,  max_iterations=1000, learning_rate=1):
    a = S.a
    b = S.b
    n = S.n
    
    # Set the initial guess for the partition

    if obj == objG:
        partition = np.linspace(a,b,n+1)
    else:
        partition = np.linspace(mOpt(S,b), mOpt(S,a), n+1)

    path = []
    path.append(partition.copy()) # add the initial guess to the path
    maxgradold = 9999

    for i in tqdm(range(max_iterations)):
        # Compute the gradient of the objective function
        grad = gradient(S, obj, partition)
        maxgradnew = np.abs(grad).max() 

        d =1
        if maxgradnew > 2 * maxgradold:
            print("   doop")
            d = maxgradold/maxgradnew
        # Adjust the partition using the gradient
        partition[1:-1] -= learning_rate * grad[1:-1] * d

        # Enforce the constraints a=g0 and gn=b or m*(b) = m0 and m*(a) = mn
        if obj == objG:
            partition[0] = a
            partition[-1] = b
        else:
            partition[0] = mOpt(S,b)
            partition[-1] =  mOpt(S,a) 

        if i > 3:
            difference = np.sum(np.abs(np.array(partition)-np.array(path[-2])))
            if difference <  1e-3 *tolerance:
                print('   ploop')
                partition = .5 * (np.array(partition)+np.array(path[-1]))

        path.append(partition.copy())

        #Check whether partition is in ascending order
        if np.all(np.diff(path) > 0) == False:
            print('unfeasible region')
            break

        # Check for convergence
        if maxgradnew < tolerance:
            break
            
        maxgradold = maxgradnew
    if allpath == 1:    
        return path
    return path[-1]

def Adam(S, obj = objG, allpath = 0, tolerance=1e-6, max_iterations=1000, learning_rate=0.01,  epsilon=1e-8,  beta1=0.9, beta2=0.999):
    """
    Adam gradient descent algorithm for optimizing the objective function
    """
    a = S.a
    b = S.b
    n = S.n
    # Set the initial guess for the partition
    if obj == objG:
        partition = np.linspace(a,b,n+1)
    else:
        partition = np.linspace(mOpt(S,b), mOpt(S,a), n+1)
    # Initialize the moment estimates for the gradient and its squared magnitude
    m = np.zeros_like(partition)
    v = np.zeros_like(partition)

    path = []
    path.append(partition.copy()) # add the initial guess to the path
    
    maxgradold = 9999

    for i in tqdm(range(max_iterations)):
        # Compute the gradient of the objective function
        grad = gradient(S, obj, partition)
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

        # Enforce the constraints a=g0 and gn=b or m*(b) = m0 and m*(a) = mn
        if obj == objG:
            partition[0] = a
            partition[-1] = b
        else:
            partition[0] = mOpt(S,b)
            partition[-1] =  mOpt(S,a) 

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
            break

        maxgradold = maxgradnew

    if allpath == 1:    
        return path
    return path[-1]

