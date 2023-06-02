# This file is dedicated to produce all results needed in my thesis. Clarity is a must!
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import root
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from Classes import *
from Settingfunctions import *

#np.random.seed(123)
N = 10000
a=1
b=10
Gammas = GammaDistributions(a,b,N) #we work with support [a=1, b=10] for gamma
def GamLabCol(mode):
    """
    mode 1: only uniform
    mode 2: normal center and normal dip
    mode 3: normal left and normal right
    """

    if mode == 1:
        return [Gammas.Eunif()], ['Uniform'] , ['blue']
    if mode == 2:
        return [Gammas.EnormC(), Gammas.EnormD()], ['Normal Center', 'Normal Dip'], ['green', 'blue']
    if mode == 3:
        return [Gammas.EnormL(), Gammas.EnormD()], ['Normal Left', 'Normal Right'], ['green', 'blue']

n=2
gammaE=Gammas.unif()         

eta = 1
s1 = S(n= n, Gamma = gammaE, eta= 1)

def my_formula(i):
    return  (a**(1-i/n))*(b**(i/n))
benchmark = np.fromfunction(my_formula,(n+1,))

def returndistributions(S):
    sigma = S.sigma
    mu = S.mu
    r = S.r
    T = S.T
    m_values = np.linspace(0.1, .5, 3)

    # Generate a standard normal distribution of outcomes
    n= 1000000
    Z = np.random.normal(size=n)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    def R(m, Z):
        return np.exp(r*T + (mu - r)*m*T-.5*(sigma**2)*(m**2)*T + m*sigma*Z*np.sqrt(T))

    # Loop over the m values and plot the distribution of R for each value
    # plot histograms for each m value
    for i, m in enumerate(m_values):
        R_values = [R(m, Z[j]) for j in range(n)]
        plt.hist(R_values, bins=500, alpha=0.5, label=f"m={m}", density=True)
        plt.axvline(np.mean(R_values), color=f'C{i}', linestyle='dashed', linewidth=5)


    # Add labels and legend to the plot
    ax.set_xlabel("R")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of R for different values of m")
    ax.legend()

    # Show the plot
    plt.show()
#returndistributions()

                

def gammadistribution(mode):
    """
    mode 1: only uniform
    mode 2: normal center and normal dip
    mode 3: normal left and normal right
    """
    x = np.linspace(a,b,101)
    
    if mode == 1:
        y0 = Gammas.unif.pdf(x)
        plt.plot(x,y0, label= 'Uniform', color= 'blue')

    if mode == 2:
        y1 = Gammas.normC.pdf(x)
        plt.plot(x,y1, label = 'Normal Center',color='orange')

        
        y4 = .5*(Gammas.normL.pdf(x) + Gammas.normR.pdf(x))
        plt.plot(x,y4, label = 'Normal Dip', color='green')

    if mode == 3:
        y2 = Gammas.normL.pdf(x)
        plt.plot(x,y2, label = 'Normal Left',color='green')

        y3 = Gammas.normR.pdf(x)
        plt.plot(x,y3, label = 'Normal Right', color= 'red')


    plt.legend(fontsize= 'large')
    plt.show()

def IntrinsicComparison(S, algorithm = GD, obj = objG):
    '''
    n: is the amount of decisions. choose n >= 3 for best graphs

    adam: 1 if use Adams algorithm, 0 if use GD algorithm
    '''
    a = S.a
    b = S.b
    for LR in [1]:
        print(f'Algorithm: {label(algorithm)}, Objective: {label(obj)}, LR={LR}')
        P= algorithm(S,obj=obj,allpath=1)
        for i in range(1,n-1):
            p_values = [(p[i],p[i+1]) for p in P]
            p1, p2 = zip(*p_values)
            plt.plot(p1, p2, '-o', label=f"Adam pair ({i},{i+1}): n={n}, LR={LR}")
    if obj == objG:
        diagonal = [(x,x) for x in np.linspace(a,b,2)]
    if obj == objM:
        diagonal = [(x,x) for x in np.linspace(trans(S,b),trans(S,a),2)]
    x1,x2 = zip(*diagonal)
    plt.plot(x1,x2,'-o', label="Feasible Boundary")
    plt.legend(fontsize=20)
    plt.show()

#IntrinsicComparison(s1, GD, objM)

def ExtrinsicComparison(S, variants, field = objG):
    """
    S: the setting class
    variants: a list of lists containing [Adam/Gd, objM/objG]
    field: whether we plot the steps in the risk or decision fields of all variants
    field = 'objG': plot in terms of risk partitions
    field = 'objM':  plot in terms of decision partitions
    """
    a = S.a
    b = S.b

    for variant in variants:
        print(f'Algorithm: {label(variant[0])}, Objective: {label(variant[1])}')
        P= variant[0](S, variant[1], allpath = 1, tolerance = 1e-3)        
        if variant[1]!=field: #translate if obj and field differ
            if variant[1] == objM:
                P = trans(S, np.array(P))
            if variant[1] == objG:
                P = trans(S, np.array(P))
        for i in range(1,n-1):
            p_values = [(p[i],p[i+1]) for p in P]
            p1, p2 = zip(*p_values)
            plt.plot(p1, p2, '-o', label=f" ({i},{i+1}):Algorithm: {label(variant[0])}, Objective: {label(variant[1])}")
    if field == objG:
        diagonal = [(x,x) for x in np.linspace(a,b,2)]
    if field == objM:
        diagonal = [(x,x) for x in np.linspace(trans(S,b),trans(S,a),2)]
    x1,x2 = zip(*diagonal)
    plt.plot(x1,x2,'-o', label="Feasible Boundary")
    plt.legend(fontsize=20)
    plt.show()
#Allvariants = [[Adam,objG],[GD,objG],[Adam, objM], [GD, objM]]
#GDvariants = [[GD, objM], [GD, objG]]
#ExtrinsicComparison(s1, Allvariants, objG)
#ExtrinsicComparison(s1, Allvariants, objM)


def consistency(S, RGamma, algorithm = GD,  obj = objG):
    S.Gamma = RGamma

    #P = algorithm(S, obj, tolerance=1e-3)
    P = benchmark
    # Plotting vertical lines at each point in P
    for point in P:
        plt.axvline(point, linewidth=1)

    samplesizes = [1e2, 1e3, 1e4]
    amount = 100
    for ss in samplesizes:
        print(f'samplesize = {ss}')
        Paths = []
        samples = RGamma.rvs((amount,int(ss)))
        for i in range(amount):
            print(f'ss={ss}, i={i}')
            EGamma =  ECDF(samples[i])
            S.Gamma =  EGamma
            P = algorithm(S, obj)[1:-1]
            if P[-1] > 10:
                print('what')
                pass
            Paths.append(P)
        plt.hist(np.ravel(np.array(Paths)), bins=500, alpha=0.5, label=f"samplesize =  {ss}")
    plt.legend()
    plt.show()

#consistency(s1, Gammas.unif(), GD, objG)

def progression(S, RGamma, algorithm = GD, obj = objG):
    """
    shows how one emperical estimation approaches a real partition over its iterations
    """
    S.Gamma = RGamma

    #Pgoal = algorithm(S,obj)
    Pgoal = benchmark
    N = 10000
    S.Gamma = ECDF(RGamma.rvs(N))
    P = algorithm(S, obj, allpath=1)
    chosenPercentages = [0, .2, .4, .6, .8]
    chosenSteps = [round(i * len(P)) for i in chosenPercentages] #note that round(i* len(P)) wil raise error
    chosenPartitions = [P[i] for i in chosenSteps]
    chosenPartitions.append(P[-2]) #append final partition
    chosenPartitions.append(Pgoal)

    circumstances = chosenPercentages.copy()
    circumstances.append(1)
    circumstances.append(1.1)

    # Plot the partitions
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(chosenPartitions)-1):
        Ry = [circumstances[i]] * len(chosenPartitions[i])
        ax.scatter(chosenPartitions[i], Ry, color='blue', s=50)
    Ry = [circumstances[len(chosenPartitions)-1]] * len(chosenPartitions[len(chosenPartitions)-1])
    ax.scatter(chosenPartitions[len(chosenPartitions)-1], Ry, color='red', s=50)
    # Set axis labels and title
    ax.set_xlabel('Estimations')
    ax.set_ylabel('Progression 0% -> 100%')
    ax.set_title('Progression of Estimated Partition')

    plt.show()     
#progression(s1, Gammas.unif(), GD, objG)

def variationGammaEta(S, mode, chosenEta = np.linspace(0, 50, 11), Fully = 0, algorithm = GD, obj = objG):
    """
    GammaSet consists of the gamma distributions to be compared such as:  [Gammas.Eunif(), Gammas.normC(), ..]
    FirstFully: 0, 1, decided whether we only take the first gamma from the GammaSet and analyse it in both graphs with the partitions and optimal elements, or we only look at the optimal decisions and optimal risk partitions of all gammas in GammaSet.
    mode 1: only uniform
    mode 2: normal center and normal dip
    mode 3: normal left and normal right
    """
    fig1, Rax = plt.subplots(figsize=(8, 6))
    fig2, Dax = plt.subplots(figsize=(8, 6))    
    
    # Define the chosen values for eta

    circumstances = chosenEta.copy()

    GammaSet, Labels, Colors = GamLabCol(mode)


    for i, gamma in enumerate(GammaSet):
        S.Gamma = gamma
        print(i, Labels[i])

        Rpartitions = [] #containing solutions without boundaries
        decisions = [] #containing the decisions

        for eta in chosenEta:
            S.eta = eta
            P = algorithm(S, obj)
            Rpartitions.append(P[1:-1])
            decisions.append([optimalDecision(S, P[j],P[j+1]) for j in range(len(P)-1)])

        Rpartitions = np.array(Rpartitions)
        decisions = np.fliplr(np.array(decisions)) #flipping decisions

        #we scatter the first eta-partitions to introduce the Labels on the agenda:
        Ry = [circumstances[0]] * len(Rpartitions[0])
        optDz = [circumstances[0]] * len(decisions[0])

        Rax.scatter(Rpartitions[0], Ry, color=Colors[i], s=50, label=Labels[i])
        Dax.scatter(decisions[0],optDz,marker='*' ,color=Colors[i], s=50, label=Labels[i])

        #And the boundaries
        Rax.scatter([a,b], [circumstances[0]]*2, color='black', s=50)
        Dax.scatter([trans(S, a),trans(S, b)], [circumstances[0]]*2, color='black', s=50)
        
        #plot the remaining etas
        for j in range(1, len(chosenEta)):
            Py = [circumstances[j]] * len(Rpartitions[j])
            optDz = [circumstances[j]] * len(decisions[j])

            Rax.scatter(Rpartitions[j], Py, color=Colors[i], s=50)
            Dax.scatter(decisions[j],optDz,marker='*' ,color=Colors[i], s=50)

            #boundaries for all eta
            Rax.scatter([a,b], [circumstances[j]]*2, color='black', s=50)
            Dax.scatter([trans(S, b),trans(S, a)], [circumstances[j]]*2, color='black', s=50)

        if Fully == 1:
            Dpartitions = trans(S, Rpartitions)
            Gdecisions =  trans(S, decisions)

            Pz = [circumstances[0]] * len(Gdecisions[0]) # y-coords of Gopt of optimal decisions
            Dy = [circumstances[0]] * len(Dpartitions[0]) # y-coords Dopt of optimal risk partitions

            #Colors[i] = 'red'

            Dax.scatter(Dpartitions[0], Dy, color=Colors[i], s=100)
            Rax.scatter(Gdecisions[0],Pz,marker='*' ,color=Colors[i], s=50)    

            for j in range(1, len(chosenEta)):
                    Pz = [circumstances[j]] * len(Gdecisions[j])
                    Rax.scatter(Gdecisions[j],Pz,marker='*' ,color=Colors[i], s=50) #add Gopt of optimal decision 

                    Dy = [circumstances[j]] * len(Dpartitions[j])
                    Dax.scatter(Dpartitions[j], Dy, color=Colors[i], s=50) #add decision partition 
            #break #break the GammaSet loop, only do this for the first one

    Rax.set_xlabel('Partitions')
    Dax.set_xlabel('Decisions')
    Rax.set_ylabel('Eta')
    Dax.set_ylabel('Eta')
    Rax.legend()
    Dax.legend()
    plt.show()
s1.n = 1
#variationGammaEta(s1, 2, np.linspace(0,2, 26), Fully=1)

      
def costOptimum(S, n, F, mode, compare = 0, algorithm = GD, obj = objG):
    """
    n: compare for m=1,..,n objective values
    f: cost function
    mode 1: only uniform
    mode 2: normal center and normal dip
    mode 3: normal left and normal right
    compare = 1,2,3 : compare gammas, etas, cost functions resp.
    """

    
    GammaSet, Labels, Colors = GamLabCol(mode)
    Etas = np.linspace(0,50,3)
    gamma = GammaSet[0]
    eta = S.eta
    optimum = 3
    indicator = np.where(compare==1, len(GammaSet),0)+np.where(compare==2, len(Etas),0) + np.where(compare==3, len(F),0)
    vary = np.linspace(-.01,.01,indicator)
    if mode ==1 and compare ==1:
        vary = np.zeros(len(GammaSet))
    
    fig1, ax1 = plt.subplots() #plot for costs at m =1,..n, where that decision is optimal
    fig2, ax2 = plt.subplots() #plot differences obj function and cost function and see intersection

    def optCostPlot(objlist, compare, f = F[0], gamma=gamma, eta = eta):
        c = [(objlist[i+1]-objlist[i])/(f(i+2)-f(i+1)) for i in range(len(objlist)-1)]
                
        X = [x for x in range(2,n)]
        Y = [[c[i],c[i+1]] for i in range(n-2)]
        for x, (y1, y2) in zip(X, Y):

            if compare ==1:
                ax1.plot([x+vary[i]] * 2, [y1, y2], '-o', color=Colors[i], label = Labels[i])
            if compare ==2:
                ax1.plot([x+vary[i]] * 2, [y1, y2], '-o', label= f'eta = {eta}')
            if compare ==3:
                ax1.plot([x+vary[i]] * 2, [y1, y2], '-o',label = costlabel(f))

        cDiff = np.diff(c)
        cPer = [cDiff[i]/c[i] for i in range(len(c)-1)] 
        return c

    def plotobj(objlist, compare, gamma=gamma, eta=S.eta):

        X = np.arange(1,n) #values n = 1,...,n-1
        diffObj = np.diff(np.array(objlist))
        if compare == 1:
            ax2.plot(X,diffObj, color=Colors[i], label = Labels[i])
        if compare == 2:
            ax2.plot(X,diffObj, label = f'eta = {S.eta}') #with corresponding objective difference values 1-2, 2-3, .. n-1-n.
        if compare == 3:
            ax2.plot(X,diffObj, color=Colors[0], label = Labels[0])

    def plotcost(optimum, c, f=F[0]):
        #plots the cost curves against an objective function with corresponding cost array c
        #n = i + 1 optimal for c in c[i], c[i+1]
        cfactor = .5*(c[optimum-1] +c[optimum-2]) 
        Nvalues = np.arange(1,n+1) #values n = 1,...,n
        costdifferences = cfactor * np.diff(f(np.array(Nvalues)))
        X = np.arange(1,n) #values n = 1,...,n-1
        ax2.plot(X, costdifferences, label = f'{costlabel(f)}, costfactor {cfactor:.5f}, optimum at {optimum}')
        if compare == 1:
            ax1.scatter(optimum + vary[i], cfactor, label = f'{Labels[i]}', s =200)
        if compare == 2:
            ax1.scatter(optimum + vary[i], cfactor, label = f'eta={eta}', s =200)
        if compare == 3:
            ax1.scatter(optimum + vary[i], cfactor, label = f'{Labels[0]}', s =200)            

    if compare == 1:
        for i, gamma in enumerate(GammaSet):
            print(Labels[i])
            eta = S.eta
            S.Gamma = gamma

            objList=np.zeros(n)
            for m in range(1,n+1):
                S.n = m
                P = algorithm(S, obj)
                value = invUsp(S, goal_function(S,P))
                objList[m-1] = value
                print(f'{m}, {value}')
            plotobj(objList, compare =1, eta = eta)
            plotcost(optimum, optCostPlot(objlist=objList, compare=1,f=F[0], gamma=gamma, eta= eta))


    if compare == 2:

        S.Gamma = GammaSet[0]
        gamma =  S.Gamma
        for i, eta in enumerate(Etas):
            S.eta = eta
            print('eta:', Etas[i])

            objList=np.zeros(n)
            for m in range(1,n+1):
                S.n = m
                P = algorithm(S, obj)
                goal =  goal_function(S,P)
                value = invUsp(S, goal)
                objList[m-1] = value
                print(f'{m}, {value}')
            plotobj(objList, compare =2)
            plotcost(optimum, optCostPlot(objlist=objList, compare=2,f=F[0], gamma=gamma, eta= S.eta))

    if compare == 3:
        S.Gamma = GammaSet[0]
        gamma =  S.Gamma
        objList=np.zeros(n)
        for m in range(1,n+1):
            S.n = m
            P = algorithm(S, obj)
            value = invUsp(S, goal_function(S,P))
            objList[m-1] = value
            print(f'{m}, {value}')
        plotobj(objList, compare =3)
        for i, f in enumerate(F):
            plotcost(optimum, optCostPlot(objlist=objList, compare=3,f=f, gamma=gamma, eta= eta),f)
    ax1.legend()
    ax2.legend()
    plt.show()
    return 
F = [lin, log, quad, exp]
costOptimum(s1, 5, F, mode =1, compare=3)
 