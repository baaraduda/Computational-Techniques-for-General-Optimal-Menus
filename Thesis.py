# This file is dedicated to produce all results needed in my thesis. Clarity is a must!
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
from functions import *


#choice of gamma distributions
N = 10000
np.random.seed(126)

#uniform
Gamma0 = stats.uniform(a,b-a)
Gamma0E = ECDF(Gamma0.rvs(N))
#plt.hist(Gamma0E.x[1:], bins=100, density=True)
#plt.show()
#normal center
Gamma1 = stats.truncnorm((a - mean) / sig, (b - mean) / sig, mean, sig)
Gamma1E = ECDF(Gamma1.rvs(N))

#normal left
Gamma2 = stats.truncnorm((a - a) / sig, (b - a) / sig, a, sig)
#Gamma2E = ECDF(Gamma2.rvs(N))

#normal right
Gamma3 = stats.truncnorm((a - b) / sig, (b - b) / sig, b, sig)
#Gamma3E = ECDF(Gamma3.rvs(N))

#normal dip

Gamma4E = ECDF(np.sort(np.concatenate([Gamma2.rvs(N//2), Gamma3.rvs(N//2)])))


#gamma = [GammaX, 0/1] where 1 is ecdf and 0 is cdf

def returndistributions():
    m_values = np.linspace(0, 1, 3)

    # Generate a standard normal distribution of outcomes
    n= 10000
    Z = np.random.normal(size=n)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    def R(m, Z):
        return np.exp(r*T + (mu - r)*m*T-.5*(sigma**2)*(m**2)*T + m*sigma*Z*np.sqrt(T))

    # Loop over the m values and plot the distribution of R for each value
    # plot histograms for each m value
    for i, m in enumerate(m_values):
        R_values = [R(m, Z[j]) for j in range(n)]
        plt.hist(R_values, bins=50, alpha=0.5, label=f"m={m}", density=True)
        plt.axvline(np.mean(R_values), color=f'C{i}', linestyle='dashed', linewidth=1)


    # Add labels and legend to the plot
    ax.set_xlabel("R")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of R for different values of m")
    ax.legend()

    # Show the plot
    plt.show()

def gammadistributions():
    x = np.linspace(a,b,101)

    y0 = Gamma0.pdf(x)
    plt.plot(x,y0, label= 'Uniform', color= 'blue')

    y1 = Gamma1.pdf(x)
    plt.plot(x,y1, label = 'Normal Center',color='orange')

    y2 = Gamma2.pdf(x)
    plt.plot(x,y2, label = 'Normal Left',color='green')

    y3 = Gamma3.pdf(x)
    plt.plot(x,y3, label = 'Normal Right', color= 'red')

    y4 = .5*(Gamma2.pdf(x) + Gamma3.pdf(x))
    plt.plot(x,y4, label = 'Normal Dip', color='green')

    plt.legend(fontsize= 'large')
    plt.show()

def comparison(n, eta, gamma, adam = 1):
    '''
    n: is the amount of decisions. choose n >= 3 for best graphs

    adam: 1 if use Adams algorithm, 0 if use GD algorithm
    '''

    for LR in [.9]:

        if adam == 1:
            print('Adam', f"LR={LR}")
            P= AdamAlgorithm(a,b, eta, gamma, guess = np.linspace(a,b,n+1))
            print(P[-2])
            for i in range(1,n-1):
                g_values = [(p[i],p[i+1]) for p in P]
                g1, g2 = zip(*g_values)
                plt.plot(g1, g2, '-o', label=f"Adam pair ({i},{i+1}): n={n}, LR={LR}")

        if adam == 0:
            print('Gradient-Descent', f"LR={LR}")
            P= GDAlgorithm(a,b, eta, gamma, guess = np.linspace(a,b,n+1))
            for i in range(1,n-1):
                g_values = [(p[i],p[i+1]) for p in P]
                g1, g2 = zip(*g_values)
                plt.plot(g1, g2, '-o', label=f"Gradient-Descent pair ({i},{i+1}): n={n}, LR={LR}")

    diagonal = [(x,x) for x in np.linspace(a,b,2)]
    x1,x2 = zip(*diagonal)
    plt.plot(x1,x2,'-o', label="Feasible Boundary")

    plt.text(b-0.1, b-0.1, "Feasible Boundary", fontsize=10)
    plt.legend(fontsize=20)
    plt.show()
#comparison(3, 1, [Gamma1E,1], 1)

def AdamGDcomparison(n,eta, gamma):
    # print('Adam', f"n={n}")
    # P= AdamAlgorithm(a,b, eta, gamma, guess = np.linspace(a,b,n+1))
    # print(P[-2])
    # for i in range(1,n-1):
    #     g_values = [(p[i],p[i+1]) for p in P]
    #     g1, g2 = zip(*g_values)
    #     plt.plot(g1, g2, '-o', label=f"Adam pair ({i},{i+1}): n={n}")

    print('Gradient-Descent', f"n={n}")
    P= GDAlgorithm(a,b, eta, gamma, guess = np.linspace(a,b,n+1))
    print(P[-2])
    for i in range(1,n-1):
        g_values = [(p[i],p[i+1]) for p in P]
        g1, g2 = zip(*g_values)
        plt.plot(g1, g2, '-o', label=f"Gradient-Descent pair ({i},{i+1}): n={n}")

    diagonal = [(x,x) for x in np.linspace(a,b,2)]
    x1,x2 = zip(*diagonal)
    plt.plot(x1,x2,'-o', label="Feasible Boundary")

    plt.text(b-0.1, b-0.1, "Feasible Boundary", fontsize=10)
    plt.legend(fontsize=20)
    plt.show()
#AdamGDcomparison(4,1,[Gamma0E,1])

def empericalEstimation(n, eta, gamma):

    P = AdamAlgorithm(a,b,eta,gamma, guess=np.linspace(a,b,n+1))
    samples = [1, 2, 3, 4]
    #for varying samplesizes, calculate the emperical partition
    chosenPartitions = [AdamAlgorithm(a,b,eta, [ECDF(gamma[0].rvs(10**i)),1], guess=np.linspace(a,b,n+1))[-2] for i in samples]
    chosenPartitions.append(P[-2])

    circumstances = samples.copy()
    circumstances.append(samples[-1]+ .1)

    # Plot the partitions
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(chosenPartitions)-1):
        Py = [circumstances[i]] * len(chosenPartitions[i])
        ax.scatter(chosenPartitions[i], Py, color='blue', s=50)
    Py = [circumstances[len(chosenPartitions)-1]] * len(chosenPartitions[len(chosenPartitions)-1])
    ax.scatter(chosenPartitions[len(chosenPartitions)-1], Py, color='red', s=50)
    # Set axis labels and title
    ax.set_xlabel('Estimations')
    ax.set_ylabel('Sample size of 10**x')
    ax.set_title('Progression of Estimated Partition')

    plt.show()
#empericalEstimation(2, 1, [Gamma0,0])

def progressionAlgorithm(n, eta, gamma):
    Pgoal = AdamAlgorithm(a,b,eta,gamma, guess=np.linspace(a,b,n+1))[-2]
    #Pgoal = np.fromfunction(my_formula,(n+1,))
    N = 1000000
    P = AdamAlgorithm(a,b,eta,[ECDF(gamma[0].rvs(N)),1], guess=np.linspace(a,b,n+1))
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
        Py = [circumstances[i]] * len(chosenPartitions[i])
        ax.scatter(chosenPartitions[i], Py, color='blue', s=50)
    Py = [circumstances[len(chosenPartitions)-1]] * len(chosenPartitions[len(chosenPartitions)-1])
    ax.scatter(chosenPartitions[len(chosenPartitions)-1], Py, color='red', s=50)
    # Set axis labels and title
    ax.set_xlabel('Estimations')
    ax.set_ylabel('Progression 0% -> 100%')
    ax.set_title('Progression of Estimated Partition')

    plt.show()     
#progressionAlgorithm(3,1,[Gamma0,0])

def varyingEta(n, Gamma):
    fig1, Pax = plt.subplots(figsize=(8, 6))
    fig2, Dax = plt.subplots(figsize=(8, 6))    
    
    # Define the chosen values for eta
    chosenEta = np.linspace(0, 200, 3)
    circumstances = chosenEta.copy()

    #choose the right label for Gamma
    #labels = ['Uniform']
    #colors = ['blue']
    labels = ['Uniform', 'Normal Center', 'Normal Dip']
    colors = ['blue', 'orange', 'green'] #
    #labels = [ 'Normal Center', 'Normal Left', 'Normal Right']
    #colors = ['orange', 'green', 'red']

    for i, gamma in enumerate(Gamma):
        print(i, labels[i])

        def m(g):
            return (mu-r)/(g*(sigma**2))

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


        solutions=[AdamAlgorithm(a,b,etai,gamma,np.linspace(a,b,n+1),max_iterations=1000)[-2] for etai in chosenEta]
        partitions=[solution[1:-1] for solution in solutions]


        decisions = np.array([[optimaldecision(solutions[i][j],solutions[i][j+1],gamma) for i in range(len(chosenEta))] for j in range(len(solutions[0])-1)]).T #

        #Gdecisions = g(decisions)
        #Dpartitions = m(np.array(partitions))

        Py = [circumstances[0]] * len(partitions[0])
        #Pz = [circumstances[0]] * len(Gdecisions[0])
        #Dy = [circumstances[0]] * len(Dpartitions[0])
        Dz = [circumstances[0]] * len(decisions[0])

        Pax.scatter(partitions[0], Py, color=colors[i], s=200, label=labels[i])
        #Dax.scatter(Dpartitions[0], Dy,marker='|', color=colors[i], s=200)

        #Pax.scatter(Gdecisions[0],Pz,marker='*' ,color=colors[i], s=50)
        Dax.scatter(decisions[0],Dz,marker='*' ,color=colors[i], s=50, label=labels[i])

        Pax.scatter([a,b], [circumstances[0]]*2, color='black', s=10)
        Dax.scatter([m(a),m(b)], [circumstances[0]]*2, color='black', s=10)
        
        for j in range(1, len(chosenEta)):
            Py = [circumstances[j]] * len(partitions[j])
            #Pz = [circumstances[j]] * len(Gdecisions[j])
            #Dy = [circumstances[j]] * len(Dpartitions[j])
            Dz = [circumstances[j]] * len(decisions[j])

            Pax.scatter(partitions[j], Py, color=colors[i], s=200)
            #Dax.scatter(Dpartitions[j], Dy, marker='|', color=colors[i], s=200)

            #Pax.scatter(Gdecisions[j],Pz,marker='*' ,color=colors[i], s=50)
            Dax.scatter(decisions[j],Dz,marker='*' ,color=colors[i], s=50)

            Pax.scatter([a,b], [circumstances[j]]*2, color='black', s=10)
            Dax.scatter([m(a),m(b)], [circumstances[j]]*2, color='black', s=10)

    Pax.set_xlabel('Partitions')
    Dax.set_xlabel('Decisions')
    Pax.set_ylabel('Eta')
    Dax.set_ylabel('Eta')
    Pax.legend()
    Dax.legend()
    plt.show()
#varyingEta(2, [[Gamma0E,1], [Gamma1E,1], [Gamma4E,1]])

def costOptimum(n,Eta,Gamma, ColLab, f):

    for i, gamma in enumerate(Gamma):




    objList=[]
    k=1
    for m in range(1,n+1):
        print(m)
        P =GDAlgorithm(a,b,eta,gamma, np.linspace(a,b,m+1))[-2]
        objList.append(goal_function(P,eta, gamma))

    objInfo = [(i+1, objList[i]) for i in range(len(objList))]
    print(objInfo)

    c = [(objList[i+1]-objList[i])/(f(i+2)-f(i+1)) for i in range(len(objList)-1)]

    cInfo = [(i+2, c[i]) for i in range(len(c))]
    print(cInfo)

    for i in range(len(c)-1):
        if c[i]<c[i+1]:
            k = i
    X = [x for x in range(2,n)]
    Y = [[c[i],c[i+1]] for i in range(n-2)]
    for x, (y1, y2) in zip(X, Y):
        plt.plot([x] * 2, [y1, y2], '-o', color='red')
    plt.xlim(0, n)  
    plt.show()

    ck = (c[k-1] + c[k])/2
    cl = (c[k] + c[k+1])/2

    Ck = [objList[i]-ck*f(i+1) for i in range(len(objList))]
    CkInfo = [(i+1, Ck[i]) for i in range(len(Ck))]


    Cl = [objList[i]-cl*f(i+1) for i in range(len(objList))]
    ClInfo = [(i+1, Cl[i]) for i in range(len(Cl))]


    print(f'cost factor c where n={k+1} is optimal is {ck} with costlist= {CkInfo}')
    print(f'cost factor c where n={k+2} is optimal is {cl} with costlist= {ClInfo}')

    # percentages = [(c[i+1]-c[i])/c[i] for i in range(len(c)-1)]
    # print(percentages)
def f(x):
    return x**2

costOptimum(10, 1, [Gamma0E,1], f)

comparison = [0, [Gamma0E, Gamma1E], ['Uniform', 'Normal Center'], ['blue', 'orange']]
