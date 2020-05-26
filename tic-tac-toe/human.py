class Human:
    def __init__(self):
        pass

    def setSymbol(self, sym):
        self.sym = sym
    
    def takeAction(self, env):
        while True:
            print('please enter position (x y):')
            move = input()
            x, y = [int(x) for x in move.split()]
            if env.isEmpty(x, y):
                env.board[env.getPos1D(x,y)] = self.sym
                break
    
    def update(self, env):
        pass
    def updateStateHistory(self, s):
        pass