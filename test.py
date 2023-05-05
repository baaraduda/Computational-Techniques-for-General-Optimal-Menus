import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import root
import scipy.stats as stats
from tqdm import tqdm
from scipy.stats import truncnorm
from scipy.stats import rv_continuous
import numpy as np

class MixtureDistribution(rv_continuous):
    def __init__(self, dist1, dist2, w1):
        self.dist1 = dist1
        self.dist2 = dist2
        self.w1 = w1
        super(MixtureDistribution, self).__init__()

    def _pdf(self, x):
        return self.w1 * self.dist1.pdf(x) + (1 - self.w1) * self.dist2.pdf(x)


a =1
b =10

#Here are the global parameters:
r = .02 #risk-free rate
mu = .05 #expected return asset
sigma = math.sqrt(.03) #volatility asset
T = 20 #time 
a=1 #lowest possible risk aversion
b=10 #highest possible risk aversion


#Here follow different possible gamma distributions on the [a,b] support with the relevant parameters.
sig= math.log(b/a) #own choice how sigma relates to a and b
mean = (a+b)/2 #own choice how the mean relates to a and b

x = np.linspace(a,b,101)

#uniform,
Gamma0 = stats.uniform(a,b)
y0 = Gamma0.pdf(x)
#plt.plot(x,y0, label= 'Uniform', color= 'blue')

#normal center
Gamma1 = stats.truncnorm((a - mean) / sig, (b - mean) / sig, mean, sig)
y1 = Gamma1.pdf(x)
#plt.plot(x,y1, label = 'Normal Center',color='orange')

#normal left
Gamma2 = stats.truncnorm((a - a) / sig, (b - a) / sig, a, sig)
y2 = Gamma2.pdf(x)
#plt.plot(x,y2, label = 'Normal Left',color='green')

#normal right
Gamma3 = stats.truncnorm((a - b) / sig, (b - b) / sig, b, sig)
y3 = Gamma3.pdf(x)
#plt.plot(x,y3, label = 'Normal Right', color= 'red')

#Normal dip
#Gamma4 = MixtureDistribution(Gamma2, Gamma3, 0.5)
y4 = .5*(Gamma2.pdf(x) + Gamma3.pdf(x))
#plt.plot(x,y4, label = 'Normal Dip', color='green')

# plt.legend(fontsize= 'large')
# plt.show()


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
