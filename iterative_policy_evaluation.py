import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4 # threshold for convergence

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

if __name__ == '__main__':
    grid = standard_grid()

    states = grid.all_states()

    V = {}
    for s in states:
        V[s] = 0
    gamma = 1.0

    while True:
        biggest_change = 0
        for s in states:
            oldV = V[s]

            # V(s) only has value if it's not a terminal state
            if s not in grid.actions:
                continue

            newV = 0
            p_a = 1.0 / len(grid.actions[s])
            for a in grid.actions[s]:
                grid.set_state(s)
                r = grid.move(a)
                newV += p_a * (r + (gamma * V[grid.current_state()]))
            V[s] = newV
            biggest_change = max(biggest_change, np.abs(oldV - newV))
        
        if biggest_change < SMALL_ENOUGH:
            break
    
    print('values for uniformly random actions:')
    print_values(V, grid)
    print('\n\n')

    # fixed policy
    policy = {
        (2,0): 'U',
        (1,0): 'U',
        (0,0): 'R',
        (0,1): 'R',
        (0,2): 'R',
        (1,2): 'R',
        (2,1): 'R',
        (2,2): 'R',
        (2,3): 'U',
    }
    print_policy(policy, grid)


    V = {}
    for s in states:
        V[s] = 0

    gamma = 0.9

    while True:
        biggest_change = 0
        for s in states:
            oldV = V[s]

            if s not in policy:
                continue

            a = policy[s]
            grid.set_state(s)
            r = grid.move(a)
            V[s] = r + gamma * V[grid.current_state()]
            biggest_change = max(biggest_change, np.abs(oldV - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break
    
    print("values for fixed policy:")
    print_values(V, grid)