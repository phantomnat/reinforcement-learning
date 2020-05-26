import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid

SMALL_ENOUGH = 10e-4 # threshold for convergence
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def print_values(V, g):
    for i in range(g.width):
        print('------------------------------------------\n')
        line = ''
        for j in range(g.height):
            v = V.get((i,j), 0)
            if v >= 0: line += ' '
            line += ' {0:.2f} '.format(v)
        print(line + '\n')
    print('------------------------------------------\n')


def print_policy(P, g):
    for i in range(g.width):
        print('------------------------------------------\n')
        line = ''
        for j in range(g.height):
            a = P.get((i,j), ' ')
            line += '  {}  |'.format(a)
        print(line + '\n')
    print('------------------------------------------\n')


# this is deterministic
# all p(s',r|s,a) = 1 or 0

if __name__ == '__main__':

    grid = negative_grid()
    # grid = standard_grid()

    print('rewards:')
    print_values(grid.rewards, grid)


    # state -> action 
    # we'll randomly
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print('policy:')
    print_policy(policy, grid)

    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state
            V[s] = 0

    # repeat until convergence - will break out when policy does not change
    while True:

        # policy evaluation step - we already know how to do this!
        while True:
            biggest_change = 0
            for s in states:
                oldV = V[s]

                if s not in policy:
                    continue
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)
                V[s] = r + (GAMMA * V[grid.current_state()])
                biggest_change = max(biggest_change, np.abs(oldV - V[s]))
            
            if biggest_change < SMALL_ENOUGH:
                break
        
        # policy improvement step
        isPolicyConverged = True
        for s in states:
            if s not in policy:
                continue
                
            oldA = policy[s]
            newA = None
            bestValue = float('-inf')
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                v = r + (GAMMA * V[grid.current_state()])
                if v > bestValue:
                    bestValue, newA = v, a
            policy[s] = newA
            if newA != oldA:
                isPolicyConverged = False
        
        if isPolicyConverged:
            break
    
    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)


