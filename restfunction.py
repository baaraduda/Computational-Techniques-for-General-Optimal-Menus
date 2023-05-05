def adam_gradient_descent(a, b, initialguess = np.linspace(a,b,n+1), learning_rate=0.1, max_iterations=100, epsilon=1e-8, tolerance=1e-2, beta1=0.9, beta2=0.999):


    """
    Adam gradient descent algorithm for optimizing the objective function
    """
    # Set the initial guess for the partition
    partition = initialguess
    
    # Initialize the moment estimates for the gradient and its squared magnitude
    m = np.zeros_like(partition)
    v = np.zeros_like(partition)

    path = []
    path.append(initialguess.copy()) # add the initial guess to the path
    
    for i in tqdm(range(max_iterations)):
        # Compute the gradient of the objective function
        grad = gradient(objective_function, partition)
        
        # Update the moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias-correct the moment estimates
        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1))
        
        # Adjust the partition using the Adam update rule
        partition[1:-1] -= learning_rate * m_hat[1:-1] / (np.sqrt(v_hat[1:-1]) + epsilon)

        # Enforce the constraints a=g0 and gn=b
        partition[0] = a
        partition[-1] = b

        path.append(partition.copy())

        # Check whether partition is in ascending order
        if np.all(np.diff(path) > 0) == False:
            print('unfeasible region')
            break

        # Check for convergence
        if np.abs(grad).max() < tolerance:
            path.append(partition.copy())
            break
            
        if i == max_iterations:
            path.append(partition.copy())
    
    return path

def gradient_descent(a, b, initialguess  , learning_rate=1, max_iterations=100, tolerance=1e-2):
    """

    """
    # Set the initial guess for the partition
    partition = initialguess

    path = []
    path.append(initialguess.copy()) # add the initial guess to the path

    for i in tqdm(range(max_iterations)):
        # Compute the gradient of the objective function
        grad = gradient(objective_function, partition)
        
        # Adjust the partition using the gradient
        partition[1:-1] -= learning_rate * grad[1:-1]

        # Enforce the constraints a=g0 and gn=b
        partition[0] = a
        partition[-1] = b

        path.append(partition.copy())

        #Check whether partition is in ascending order
        if np.all(np.diff(path) > 0) == False:
            break

        # Check for convergence
        if np.abs(grad).max() < tolerance:
            path.append(partition.copy())
            break
            
        if i == max_iterations:
            path.append(partition.copy())

    return path

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


def nOpt(a, b, N, k, n):
    """ simulates optimal decision m for social planner over a partition centered around the hypothetical optimum
    
    Arguments
    ---------
    a,b : support distribution of risk preferences gamma
    N : how many simulations to estimate mean
    k: how wide partition
    n: how dense partition
    """    
    optimalm = optimaldecision(a,b)
    m_values = np.linspace(optimalm-k, optimalm+k, n)


    Uspmean = []
    for m in m_values:
        #sample N from distribution Gamma
        G = Gamma.rvs(N)
        ceG = np.exp(r*T + (mu - r)*m*T-.5*G*(sigma**2)*(m**2)*T) #calculate CE per risk type
        Uspmean.append(np.mean(Usp(eta, ceG)))   #for given m, this is estimated mean of SP utility
    plt.plot(m_values, Uspmean, label='Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    margmax = np.argmax(Uspmean)*2*k/(n-1) + optimalm-k

    return 'argmax at', margmax, "and optimal m:", optimalm

def Uin(gamma, R):
    """returns utility investor for given gamma and return R"""  
    if gamma == 1:
        return np.log(R)
    else:
        return (R**(1-gamma)-1)/(1-gamma)

def Usp(eta,c):
    """returns utility social planner for given eta and certainty equivalent c""" 
    if eta == 1:
        return np.log(c)
    else:
        return (c**(1-eta)-1)/(1-eta)

def R(m, Z):
    """returns return R for given decision m and outcome standard normal distribution Z""" 
    return np.exp(r*T + (mu - r)*m*T-.5*(sigma**2)*(m**2)*T + m*sigma*Z*math.sqrt(T))

def singleOpt(N, k, n):
    """
    simulates optimal decision m for single investor over a partition centered around the hypothetical optimum

    Arguments
    ---------
    N: how many simulation per partition element to estimate the mean
    k: how wide partition
    n: how dense partition
    """ 
    merton = (mu-r)/(gamma*(sigma**2))

    m_values = np.linspace(merton-k, merton+k, n)

    Uinmean = []

    #simulate N standard normal variables


    for m in m_values:
        Z = np.random.randn(N)
        RZ = R(m, Z)
        Uinmean.append(np.mean(Uin(gamma, RZ))) 

    plt.plot(m_values, Uinmean, label='Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    margmax = np.argmax(Uinmean)*2*k/n + merton-k

    return 'argmax at', margmax, "and merton fraction:", merton,

# partitions = []
# guesses = [np.fromfunction(my_formula,(n+1,))]
# solution = AdamAlgorithm(a, b, eta=chosenEta[0], Gamma=gamma, guess=guesses[0], learning_rate=.1)[-2]
# partitions.append(solution[1:-1])
# guesses.append(solution)

# for ideta, eta in enumerate(chosenEta, start=1):
#     solution=AdamAlgorithm(a, b, eta=eta, Gamma=gamma, guess=guesses[ideta], learning_rate=.1)[-2]
#     partitions.append(solution[1:-1])
#     guesses.append(solution)