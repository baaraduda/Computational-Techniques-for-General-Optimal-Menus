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



