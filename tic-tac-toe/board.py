import numpy as np

from human import Human

def getStateHashAndWinner(env, x, y):
    pos1d = (y*3)+x

    results = []

    for v in (0, env.x, env.o):
        env.board[pos1d] = v
        if x == 2:
            if y == 2:
                # board is full
                state = env.getState()
                ended = env.gameOver(forceCalc=True)
                winner = env.getWinner()
                results.append((state, winner, ended))
            else:
                results += getStateHashAndWinner(env, 0, y+1)
        else:
            results += getStateHashAndWinner(env, x+1, y)

    return results

def initialV(states, playerSymbol=-1):
    V = np.zeros(len(states))
    for state, winner, ended in states:
        if ended:
            if playerSymbol == winner:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V

class Player:
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.stateHistory = []

    def setV(self, V):
        self.V = V
    def resetHistory(self):
        self.stateHistory = []
    def setSymbol(self, sym):
        self.sym = sym
    def setVerbose(self, v):
        self.verbose = v

    def takeAction(self, env):
        r = np.random.rand()
        bestRate = None
        if r < self.eps:
            # take random action
            if self.verbose:
                print("taking a random action")
            possibleMoves = []
            for y in range(3):
                for x in range(3):
                    if env.isEmpty(x, y):
                        possibleMoves.append((x,y))
            idx = np.random.choice(len(possibleMoves))
            nextMove = possibleMoves[idx]
        else:
            nextMove = None
            bestValue = -1
            pos2Value = {}
            for y in range(3):
                for x in range(3):
                    if not env.isEmpty(x,y):
                        continue
                    # what is the state if we made this move?
                    env.board[env.getPos1D(x,y)] = self.sym
                    state = env.getState()
                    env.board[env.getPos1D(x,y)] = 0
                    pos2Value[(x,y)] = self.V[state]
                    if self.V[state] > bestValue:
                        bestValue = self.V[state]
                        bestState = state
                        nextMove = (x,y)
            if self.verbose:
                print("taking a greedy action")
                print('')
                for y in range(3):
                    texts = []
                    for x in range(3):
                        if env.isEmpty(x,y):
                            texts.append('{:.2f}'.format(pos2Value[(x,y)]))
                        else:
                            pos1D = env.getPos1D(x,y)
                            sym = '  '
                            if env.board[pos1D] == env.x:
                                sym = 'X '
                            elif env.board[pos1D] == env.o:
                                sym = 'O '
                            texts.append(sym)
                    print(' | '.join(texts))
                    print('-'*9)
                print('')
                print('-'*9)
                print('')
        
        env.board[env.getPos1D(nextMove[0],nextMove[1])] = self.sym

    def updateStateHistory(self, state):
        self.stateHistory.append(state)

    def update(self, env):
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.stateHistory):
            v = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = v
            target = v
        self.resetHistory()

class Environment:
    def __init__(self):
        self.board = np.zeros(9)
        self.turn = 0
        self.o = -1    # p1
        self.x = 1    # p2
        self.winner = None
        self.ended = False
        self.numStates = 3 ** (3*3)

    def getPos1D(self, x, y):
        return (y*3)+x

    def isEmpty(self, x, y):
        return self.board[self.getPos1D(x, y)] == 0

    def reward(self, sym):
        if not self.gameOver():
            return 0
        return 1 if self.winner == sym else 0

    def numToSymbol(self, board):
        symbols = []
        for b in board:
            if b == self.o:
                symbols.append('O')
            elif b == self.x:
                symbols.append('X')
            else:
                symbols.append(' ')
        return symbols
    
    def drawBoard(self):
        print(' | '.join(self.numToSymbol(self.board[:3])))
        print('-' * 9)
        print(' | '.join(self.numToSymbol(self.board[3:6])))
        print('-' * 9)
        print(' | '.join(self.numToSymbol(self.board[6:])))
        print(' ' * 9)

    def isGameOver(self):
        return self.ended

    def getState(self):
        h = 0
        for i,c in enumerate(self.board):
            v = 0
            if c == self.o:
                v = 1
            elif c == self.x:
                v = 2
            h += (3 ** i) * v
        return h

    def gameOver(self, forceCalc=False):
        if not forceCalc and self.ended:
            return self.ended
    
        self.ended = False
        self.winner = None

        winnerSymbol = ''
        # find winner 
        if self.board[0]:
            if (self.board[0] == self.board[1] and self.board[0] == self.board[2]) or \
                (self.board[0] == self.board[3] and self.board[0] == self.board[6]) or \
                (self.board[0] == self.board[4] and self.board[0] == self.board[8]):
                winnerSymbol = self.board[0]
        if self.board[1] and (self.board[1] == self.board[4] and self.board[1] == self.board[7]):
            winnerSymbol = self.board[1]
        if self.board[2]:
            if (self.board[2] == self.board[5] and self.board[2] == self.board[8]) or \
                (self.board[2] == self.board[4] and self.board[2] == self.board[6]):
                winnerSymbol = self.board[2]
        if self.board[3] and (self.board[3] == self.board[4] and self.board[3] == self.board[5]):
            winnerSymbol = self.board[3]
        if self.board[6] and (self.board[6] == self.board[7] and self.board[6] == self.board[8]):
            winnerSymbol = self.board[6]
        
        if winnerSymbol:
            self.winner = winnerSymbol
        if not self.ended and self.winner:
            self.ended = True

        # check if draw
        if np.all((self.board == 0) == False):
            self.winner = None
            self.ended = True
    
        # if self.ended and not self.winner:
        #     self.drawBoard()
        #     print('ended:', self.ended, 'winner:', self.winner)

        return self.ended

    def getWinner(self):
        return self.winner

def playGame(p1: Player, p2: Player, env: Environment, draw=False):
    currentPlayer = None

    while not env.gameOver():
        currentPlayer = p2 if currentPlayer == p1 else p1

        if draw:
            if draw == 1 and currentPlayer == p1:
                env.drawBoard()
            if draw == 2 and currentPlayer == p2:
                env.drawBoard()
        
        currentPlayer.takeAction(env)
        state = env.getState()
        p1.updateStateHistory(state)
        p2.updateStateHistory(state)
    
    if draw:
        env.drawBoard()

    p1.update(env)
    p2.update(env)



# e.drawBoard()
# e.place(p1, 1,1)
# e.place(p2, 0,0)
# e.drawBoard()


# results = getStateHashAndWinner(Environment(None, None), 0, 0)
# print(results)

if __name__ == '__main__':
    p1 = Player()
    p2 = Player()

    env = Environment()
    states = getStateHashAndWinner(env, 0, 0)
    v1 = initialV(states, env.o)
    p1.setV(v1)
    v2 = initialV(states, env.x)
    p2.setV(v2)

    p1.setSymbol(env.o)
    p2.setSymbol(env.x)

    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        playGame(p1, p2, Environment())
    
    human = Human()
    human.setSymbol(env.x)
    p1.setVerbose(True)
    p2.setVerbose(True)
    playGame(p1, human, Environment(), draw=2)
    human.setSymbol(env.o)
    playGame(human, p2, Environment(), draw=1)