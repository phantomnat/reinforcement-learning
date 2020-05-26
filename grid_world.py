import numpy as numpy
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)
    
    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        
        # return a reward (if any)
        return self.rewards.get((self.i, self.j), 0)
    
    def undo_move(self, action):
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        
        assert(self.current_state() in self.all_states())
    
    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions
    
    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()).union(set(self.rewards.keys()))

def standard_grid():
    #
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    #
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0,0): set('DR'),
        (0,1): set('LR'),
        (0,2): set('LDR'),
        (1,0): set('UD'),
        (1,2): set('UDR'),
        (2,0): set('UR'),
        (2,1): set('LR'),
        (2,2): set('LRU'),
        (2,3): set('LU'),
    }
    g.set(rewards, actions)
    return g

def negative_grid(step_cost=-0.1):
    # in this game we want to try to minimize the number of moves
    # so we will penalize every move
    g = standard_grid()
    g.rewards.update({
        (0,0): step_cost,
        (0,1): step_cost,
        (0,2): step_cost,
        (1,0): step_cost,
        (1,2): step_cost,
        (2,0): step_cost,
        (2,1): step_cost,
        (2,2): step_cost,
        (2,3): step_cost,
    })
    return g

def play_game(agent, env):
    pass
