import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import run_experiment as run_experiment_eps
from optimistic_initial import run_experiment as run_experiment_oi
import math

class Bandit:
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.mean = 0
        # self.N = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m
    
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x

def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in range(1,N+1):
        j = np.argmax([(b.mean + math.sqrt((2 * math.log(i)) / (b.N+0.01))) for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i-1] = x

    cumulative_average = np.cumsum(data) / (np.arange(N)+1)

    # plot moving average ctr
    # plt.plot(cumulative_average)
    # plt.plot(np.ones(N)*m1)
    # plt.plot(np.ones(N)*m2)
    # plt.plot(np.ones(N)*m3)
    # plt.xscale('log')
    # plt.show()

    for b in bandits:
        print(b.mean)
    
    return cumulative_average

if __name__ == "__main__":
    c_eps = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
    c_oi = run_experiment_oi(1.0, 2.0, 3.0, 100000)
    c_ucb1 = run_experiment(1.0, 2.0, 3.0, 100000)

    # log scale plot
    plt.plot(c_eps, label='eps = 0.1')
    plt.plot(c_oi, label='oi')
    plt.plot(c_ucb1, label='ucb1')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_eps, label='eps = 0.1')
    plt.plot(c_oi, label='oi')
    plt.plot(c_ucb1, label='ucb1')
    plt.legend()
    plt.show()
