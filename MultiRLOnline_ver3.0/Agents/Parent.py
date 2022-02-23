import numpy as np

class Parent():
    def __init__(self, stateLength1, stateLength2, action1Length, action2Length, parentActionLength, unitTransshipmentRevenue, unitTransshipmentCost):
        self.unitTransshipmentRevenue = unitTransshipmentRevenue  # 38     
        self.unitTransshipmentCost = unitTransshipmentCost    # 2, cost of relocating the inventory from i to j
        self.q_balanced = np.zeros((stateLength1, stateLength2, action1Length, action2Length, parentActionLength))      # initialize to zeros
        self.transshipmentAmount = (-9999, -9999)
        self.transshipmentAmountList = []
        
        self.q_balanced_new = np.zeros((stateLength1, stateLength2, parentActionLength))
        self.q_cnt = np.zeros((stateLength1, stateLength2, parentActionLength))