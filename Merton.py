import numpy as np
import matplotlib.pyplot as plt
from functions import *

def comparisonAdam(n):
    tolerance = 1e-3
    print('adam', f"n={n}")
    for etai in np.linspace(0,100,5):
        P= AdamAlgorithm(a,b, eta = etai, Gamma= [Gamma3], guess = np.linspace(a,b,n+1), learning_rate=.1, max_iterations=1000, epsilon=1e-8, tolerance=tolerance, beta1=0.9, beta2=0.999)
        for i in range(1,n-1):
            g_values = [(p[i],p[i+1]) for p in P]
            g1, g2 = zip(*g_values)
            plt.plot(g1, g2, '-o', label=f"Adam pair ({i},{i+1}): n={n}, eta={etai}")

    # print('normal', f"n={n}")
    # P= Algorithm(a,b,eta =1, Gamma = Gamma0, guess=np.linspace(a,b,n+1), learning_rate=1, max_iterations=1000,tolerance=tolerance)
    # for i in range(1,n-1):
    #     g_values = [(p[i],p[i+1]) for p in P]
    #     g1, g2 = zip(*g_values)
    #     plt.plot(g1, g2, '-o', label=f"Normal pair ({i},{i+1}): n={n}")
    diagonal = [(x,x) for x in np.linspace(a,b,2)]
    x1,x2 = zip(*diagonal)
    plt.plot(x1,x2,'-o', label="Feasible Boundary")

    plt.text(b-0.1, b-0.1, "Feasible Boundary", fontsize=10)
    plt.legend(fontsize=20)
    plt.show()

def progression(P):
    chosenPercentages = [0, .2, .4, .6, .8]
    chosenSteps = [round(i * len(P)) for i in chosenPercentages] #note that round(i* len(P)) wil raise error
    chosenPartitions = [P[i] for i in chosenSteps]
    chosenPartitions.append(P[-2]) #append final partition
    def my_formula(i):
        return  (a**(1-i/n))*(b**(i/n))
    benchmark = np.fromfunction(my_formula,(n+1,))
    chosenPartitions.append(benchmark)

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

    estimation = P[-2]
    Gamma=[Gamma0]
    print('benchmark:', benchmark, 'with value:', objective_function(benchmark,Gamma), "and goal value:", goal_function(benchmark, Gamma))
    print('improvement from benchmark:', (goal_function(estimation,Gamma) - goal_function(benchmark,Gamma)) / goal_function(benchmark,Gamma)* 100, '%,  ')
    print('estimation:' , estimation , 'with value:' , objective_function(estimation, Gamma), "and goal value:", goal_function(estimation, Gamma))
    print('value guess: ', objective_function(guessP,Gamma), 'improvement from guess', (goal_function(estimation,Gamma) - goal_function(guessP,Gamma)) / goal_function(guessP,Gamma) * 100, '%')


def varyingeta(n, gammas, labels, colors):
    # Initialize figures
    fig1, Pax = plt.subplots(figsize=(8, 6))
    fig2, Dax = plt.subplots(figsize=(8, 6))
    
    # Define the chosen values for eta
    chosenEta = np.linspace(0, 200, 5)
    circumstances = chosenEta.copy()

    # Iterate over gammas and compute the partitions for each eta value

    for i, gamma in enumerate(gammas):
        print(i, labels[i])

        def g(m):
            return (mu-r)/(m*(sigma**2))
        
        def m(g):
            return (mu-r)/(g*(sigma**2))

        def optimaldecision(a,b,Gamma):
        
            def f(m):
                def e(g):
                    return np.exp(g*.5*(sigma**2)*(eta-1)*T*(m**2))
                if len(Gamma)==1:
                    integrand1 = lambda g: g * e(g) * Gamma[0].pdf(g)
                    E1,_ = integrate.quad(integrand1, a, b)
                    integrand2 = lambda g: e(g) * Gamma[0].pdf(g)
                    E2,_ = integrate.quad(integrand2, a, b)

                    return m-(mu-r)/((sigma**2)*E1/E2)
                
                if len(Gamma)==2:
                    integrand1 = lambda g: g * e(g) * .5 * (Gamma[0].pdf(g) + Gamma[1].pdf(g))
                    E1,_ = integrate.quad(integrand1, a, b)
                    integrand2 = lambda g: e(g) * .5 * (Gamma[0].pdf(g) + Gamma[1].pdf(g))
                    E2,_ = integrate.quad(integrand2, a, b)

                    return m-(mu-r)/((sigma**2)*E1/E2)
        
            x0 = .5 * ((mu-r)/(b*(sigma**2) + (mu-r)/(a*(sigma**2))))

            sol = root(f, x0)

            return sol.x[0]

        solutions=[AdamAlgorithm(a,b,etai,gamma,np.linspace(a,b,n+1),max_iterations=1000 ,tolerance=1e-6)[-2] for etai in chosenEta]
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

    '''
        partitions = []
        solutions = []
        guesses = [np.fromfunction(my_formula,(n+1,))]
        solution = AdamAlgorithm(a, b, eta=chosenEta[0], Gamma=gamma, guess=guesses[0], tolerance=1e-2)[-2]

        solutions.append(solution)
        partitions.append(solution[1:-1])
        guesses.append(solution)

        for ideta, eta in enumerate(chosenEta, start=1):
            solution=AdamAlgorithm(a, b, eta=eta, Gamma=gamma, guess=guesses[ideta], tolerance=1e-2)[-2]
            partitions.append(solution[1:-1])
            guesses.append(solution)
            solutions.append(solution)
    '''
    
    # Set axis labels and titles
    Pax.set_xlabel('Partitions')
    Dax.set_xlabel('Decisions')
    Pax.set_ylabel('Eta')
    Dax.set_ylabel('Eta')
    Pax.legend()
    Dax.legend()
    plt.show()




eta= 1
n=5

learning_rate = 1
max_iterations = 100
tolerance = 1e-1

epsilon= 1e-8

#uniform,
Gamma0 = stats.uniform(a,b)

#normal center
Gamma1 = stats.truncnorm((a - mean) / sig, (b - mean) / sig, mean, sig)

#normal left
Gamma2 = stats.truncnorm((a - a) / sig, (b - a) / sig, a, sig)

#normal right
Gamma3 = stats.truncnorm((a - b) / sig, (b - b) / sig, b, sig)


# Define the gammas to use
gammas0 = [[Gamma3]]
labels0 = ['Uniform']
colors0 = ['blue']

gammas1 = [[Gamma0], [Gamma1],[Gamma2, Gamma3]] #
labels1 = ['Uniform', 'Normal Center', 'Normal Dip'] #
colors1 = ['blue', 'orange', 'green'] #

gammas2 = [[Gamma1], [Gamma2], [Gamma3]]
labels2 = [ 'Normal Center', 'Normal Left', 'Normal Right']
colors2 = ['orange', 'green', 'red']

# comparisonAdam(3)
# comparisonAdam(4)
# comparisonAdam(5)

# P = AdamAlgorithm(a,b,eta=1, Gamma=[Gamma0], guess=np.linspace(a,b,n+1) ,tolerance=1e-3)
# progression(P)
varyingeta(2, gammas1, labels1, colors1)
#comparisonAdam(3)




