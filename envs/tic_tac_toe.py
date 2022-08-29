import numpy as np

class TicTacToe:
    def __init__(self):
        self.state = np.zeros((3,3), dtype=int)
        self.player = 1
        self.winner = None
        self.done = False

    def step(self, action):
        self.state[action] = self.player
        self.player = -self.player
        self.winner = self.get_winner(self.state)
        self.done = self.winner is not None or len(self.get_legal_actions(self.state)) == 0
        return self.state, self.winner, self.done

    def get_winner(self, state):
        for row in state:
            if abs(np.sum(row)) == len(row):
                return np.sum(row) / len(row)
        for col in state.transpose():
            if abs(np.sum(col)) == len(col):
                return np.sum(col) / len(col)
        if abs(np.sum(np.diag(state))) == len(np.diag(state)):
            return np.sum(np.diag(state)) / len(np.diag(state))
        if abs(np.sum(np.diag(np.fliplr(state)))) == len(np.diag(np.fliplr(state))):
            return np.sum(np.diag(np.fliplr(state))) / len(np.diag(np.fliplr(state)))
        
        if np.prod(state) != 0:
            return 0
                
        return None

    def get_legal_actions(self, state):
        return np.argwhere(state == 0)

    def reset(self):
        self.state = np.zeros((3,3), dtype=int)
        self.player = 1
        self.winner = None
        self.done = False
        return self.state
        
