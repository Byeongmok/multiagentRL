import numpy as np

class States():
    def __init__(self):
        
        self.tempInt = np.array([0, 10, 20, 30, 40, 50, 60, 80, 100])

        
        self.state = []
        
        for i in range(self.tempInt.size):
            line = []
            for j in range(self.tempInt.size):
                line.append([self.tempInt[i], self.tempInt[j]])
            self.state.append(line)
        
        self.state = tuple(self.state)