from Agents.Parent import Parent
import numpy as np

class Retailer(Parent):
    def __init__(self, initialInventory, stateLength1, stateLength2, actionLength, parentActionLength, xLength, type, unitTransshipmentRevenue, unitTransshipmentCost, 
                 unitHoldingCost, unitUnmetDemandPenalty):
        super(Retailer, self).__init__(stateLength1, stateLength2, actionLength, actionLength, parentActionLength, unitTransshipmentRevenue, unitTransshipmentCost)
        self.inventoryLevel = initialInventory
        self.unitRevenue = 26.0         # 26
        self.unitOrderingCost = 10.0    # 10 
        self.unitHoldingCost = unitHoldingCost  # 1 (0.2 였음 하지만 1로 하기로 함)
        self.unitPenaltyforUnmetDemand = unitUnmetDemandPenalty   # 26이 였음
        self.orderAmount = -9999
        self.xAmount = -9999
        self.yAmount = -9999
        self.yAmount_estimate = -9999
        
        self.realizedRewardList = []
        self.unmetDemandList = []
        self.amountTransToList = []
        self.inventoryLevelBeginningList = []
        self.inventoryLevelRemainingList = []
        self.orderAmountList = []
        self.yAmountList = []
        self.D = [] # thisState, anotherState, another x
        
        self.chromosome = []
        
        
        if type == 0:   # balanced gqp q-learning
            self.q_balanced = np.zeros((stateLength1, stateLength2, actionLength, parentActionLength))      # initialize to zeros
                
        if type == 1:   # decentralized q-learning (trans X)
            self.q_decentralized = np.zeros((stateLength1, actionLength))                                   # initialize to zeros
            self.q_cnt = np.zeros((stateLength1, actionLength))
        
        
        if type == 5 or type == 6 or type == 7:   # balanced gap new q-learning
            self.q_balanced_new = np.zeros((stateLength1, stateLength2, actionLength, parentActionLength))
            self.q_cnt = np.zeros((stateLength1, stateLength2, actionLength))
    
    
    def GetX2hatIndex(self, thisCurrentStateIndex1, anotherCurrentStateIndex):
        sum1 = 0.0
        cnt1 = 0
        sum2 = 0.0
        cnt2 = 0
        for i in range(len(self.D)):
            if self.D[i][0] == thisCurrentStateIndex1 and self.D[i][1] == anotherCurrentStateIndex:
                sum1 += self.D[i][2]
                cnt1 += 1
            if self.D[i][1] == anotherCurrentStateIndex:
                sum2 += self.D[i][2]
                cnt2 += 1
                
        if sum1 != 0.0:
            return round(sum1/cnt1)
        elif sum2 != 0.0:
            return round(sum2/cnt2)
        else:
            return 0
        
    
    def InventoryUpdate_balancedGap(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, anotherInventoryLevel):
        # balanced gap algorithm
        self.inventoryLevel = max(self.inventoryLevel 
                                    + max(self.orderAmount - realizedSupplyLoss, 0)
                                    + min(anotherInventoryLevel, getTransshipmentFrom)
                                    - min(self.inventoryLevel, sendTransshipmentTo)
                                    - thisRealizedDemand, 0)

            
    def GetInventoryUpdate_balancedGap(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, anotherInventoryLevel):
        # balanced gap algorithm
        inventoryLevel = max(self.inventoryLevel 
                                    + max(self.orderAmount - realizedSupplyLoss, 0)
                                    + min(anotherInventoryLevel, getTransshipmentFrom)
                                    - min(self.inventoryLevel, sendTransshipmentTo)
                                    - thisRealizedDemand, 0)            
            
        return inventoryLevel
    
    
    
    
    def InventoryUpdate_balancedGap_new(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, anotherInventoryLevel):
        # balanced gap algorithm
        self.inventoryLevel = max(self.inventoryLevel 
                                    + max(self.orderAmount - realizedSupplyLoss, 0)
                                    + min(anotherInventoryLevel, getTransshipmentFrom)
                                    - min(self.inventoryLevel, sendTransshipmentTo)
                                    - thisRealizedDemand, 0)

            
    def GetInventoryUpdate_balancedGap_new(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, anotherInventoryLevel):
        # balanced gap algorithm
        inventoryLevel = max(self.inventoryLevel 
                                    + max(self.orderAmount - realizedSupplyLoss, 0)
                                    + min(anotherInventoryLevel, getTransshipmentFrom)
                                    - min(self.inventoryLevel, sendTransshipmentTo)
                                    - thisRealizedDemand, 0)            
            
        return inventoryLevel
    
    
    def InventoryUpdate_decentralized(self, thisRealizedDemand, realizedSupplyLoss):
        # decentralized algorithm
        self.inventoryLevel = max(self.inventoryLevel 
                                    + max(self.orderAmount - realizedSupplyLoss, 0)
                                    - thisRealizedDemand, 0)
    
    def GetInventoryUpdate_decentralized(self, thisRealizedDemand, realizedSupplyLoss):
        # decentralized algorithm
        inventoryLevel = max(self.inventoryLevel 
                                    + max(self.orderAmount - realizedSupplyLoss, 0)
                                    - thisRealizedDemand, 0)
        
        return inventoryLevel
    
    
    def InventoryUpdate_centralized(self, realizedDemand, realizedSupplyLoss, y_ji):
        # centralized algorithm
        self.inventoryLevel = max(self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + y_ji - self.yAmount - realizedDemand, 0)
        
    
    
    def GetInventoryUpdate_centralized(self, realizedDemand, realizedSupplyLoss, y_ji):
        # centralized algorithm
        inventoryLevel = max(self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + y_ji - self.yAmount - realizedDemand, 0)
        return inventoryLevel
        
    
    
    def InventoryUpdate_centralized_new(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, anotherInventoryLevel):
        # centralized algorithm
        self.inventoryLevel = max(self.inventoryLevel 
                                    + max(self.orderAmount - realizedSupplyLoss, 0)
                                    + min(anotherInventoryLevel, getTransshipmentFrom)
                                    - min(self.inventoryLevel, sendTransshipmentTo)
                                    - thisRealizedDemand, 0)

            
    def GetInventoryUpdate_centralized_new(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, anotherInventoryLevel):
        # centralized algorithm
        inventoryLevel = max(self.inventoryLevel 
                                    + max(self.orderAmount - realizedSupplyLoss, 0)
                                    + min(anotherInventoryLevel, getTransshipmentFrom)
                                    - min(self.inventoryLevel, sendTransshipmentTo)
                                    - thisRealizedDemand, 0)            
            
        return inventoryLevel
    
    
    def GetRealizedReward_balancedGap(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, unitTransRevenueAnotherRetailer, anotherInventoryLevel):
        realizedReward = ((self.unitRevenue * min(thisRealizedDemand, self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitOrderingCost * self.orderAmount) 
                          - (self.unitHoldingCost * (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitPenaltyforUnmetDemand * max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)) 
                          - (unitTransRevenueAnotherRetailer * (min(anotherInventoryLevel, getTransshipmentFrom))) 
                          - (self.unitTransshipmentCost * min(self.inventoryLevel, sendTransshipmentTo)) 
                          + (self.unitTransshipmentRevenue * min(self.inventoryLevel, sendTransshipmentTo)))
        
        return realizedReward
    
    def GetUnmetDemand_balancedGap(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, anotherInventoryLevel):
        unmetDemand = max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)
        
        return unmetDemand
    
    
    def GetRealizedReward_balancedGap_new(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, unitTransRevenueAnotherRetailer, anotherInventoryLevel):
        realizedReward = ((self.unitRevenue * min(thisRealizedDemand, self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitOrderingCost * self.orderAmount) 
                          # - (self.unitHoldingCost * (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitHoldingCost * self.inventoryLevel) 
                          - (self.unitPenaltyforUnmetDemand * max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)) 
                          - (unitTransRevenueAnotherRetailer * (min(anotherInventoryLevel, getTransshipmentFrom))) 
                          - (self.unitTransshipmentCost * min(self.inventoryLevel, sendTransshipmentTo)) 
                          + (self.unitTransshipmentRevenue * min(self.inventoryLevel, sendTransshipmentTo)))
        
        return realizedReward
    
    
    
    def GetUnmetDemand_balancedGap_new(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, unitTransRevenueAnotherRetailer, anotherInventoryLevel):
        unmetDemand = max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)
        
        return unmetDemand
    
    
    
    def GetRealizedReward_balancedGap_new_online(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, unitTransRevenueAnotherRetailer, anotherInventoryLevel):
        realizedReward = ((self.unitRevenue * min(thisRealizedDemand, self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitOrderingCost * self.orderAmount) 
                          - (self.unitHoldingCost * (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitPenaltyforUnmetDemand * max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)) 
                          - (unitTransRevenueAnotherRetailer * (min(anotherInventoryLevel, getTransshipmentFrom))) 
                          - (self.unitTransshipmentCost * min(self.inventoryLevel, sendTransshipmentTo)) 
                          + (self.unitTransshipmentRevenue * min(self.inventoryLevel, sendTransshipmentTo)))
        
        unmetDemand = max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)
        
        return realizedReward, unmetDemand
    
    def GetRealizedReward_balancedGap_new_online_Tilt(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, unitTransRevenueAnotherRetailer, anotherInventoryLevel, beta):
        realizedReward = ((self.unitRevenue * min(thisRealizedDemand, self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitOrderingCost * self.orderAmount) 
                          - (self.unitHoldingCost * (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitPenaltyforUnmetDemand * beta * max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)) 
                          - (unitTransRevenueAnotherRetailer * (min(anotherInventoryLevel, getTransshipmentFrom))) 
                          - (self.unitTransshipmentCost * min(self.inventoryLevel, sendTransshipmentTo)) 
                          + (self.unitTransshipmentRevenue * min(self.inventoryLevel, sendTransshipmentTo)))
        
        return realizedReward
    
    
    def GetRealizedReward_decentralized(self, thisRealizedDemand, realizedSupplyLoss):
        realizedReward = ((self.unitRevenue * min(thisRealizedDemand, self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0))) 
                          - (self.unitOrderingCost * self.orderAmount) 
                          # - (self.unitHoldingCost * (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0)))
                          - (self.unitHoldingCost * self.inventoryLevel)
                          - (self.unitPenaltyforUnmetDemand * max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0)), 0)))
        
        return realizedReward
    
    
    def GetUnmetDemand_decentralized(self, thisRealizedDemand, realizedSupplyLoss):
        unmetDemand = max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0)), 0)
        
        return unmetDemand
    
    
    def GetRealizedReward_centralized(self, realizedDemand, realizedSupplyLoss, anotherRetailer):
        # centralized q-learning
        # for a single reatiler
        
        realizedReward = ((self.unitRevenue * min(realizedDemand, self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + anotherRetailer.yAmount - self.yAmount)) 
                          - (self.unitOrderingCost * self.orderAmount)
                          - (self.unitHoldingCost * (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + anotherRetailer.yAmount - self.yAmount)) 
                          - (self.unitPenaltyforUnmetDemand * max(realizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + anotherRetailer.yAmount - self.yAmount), 0))
                          - (self.unitTransshipmentCost * self.yAmount)) 
        
        return realizedReward
    
    
    def GetUnmetDemand_centralized(self, realizedDemand, realizedSupplyLoss, anotherRetailer):
        # centralized q-learning
        # for a single reatiler
        
        unmetDemand = max(realizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + anotherRetailer.yAmount - self.yAmount), 0)
        
        return unmetDemand
    
    
    
    def GetRealizedReward_centralized_new(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, unitTransRevenueAnotherRetailer, anotherInventoryLevel):
        realizedReward = ((self.unitRevenue * min(thisRealizedDemand, self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitOrderingCost * self.orderAmount) 
                          # - (self.unitHoldingCost * (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo))) 
                          - (self.unitHoldingCost * self.inventoryLevel) 
                          - (self.unitPenaltyforUnmetDemand * max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)) 
                          - (unitTransRevenueAnotherRetailer * (min(anotherInventoryLevel, getTransshipmentFrom))) 
                          - (self.unitTransshipmentCost * min(self.inventoryLevel, sendTransshipmentTo)) 
                          + (self.unitTransshipmentRevenue * min(self.inventoryLevel, sendTransshipmentTo)))
        
        return realizedReward
    
    
    def GetUnmetDemand_centralized_new(self, thisRealizedDemand, realizedSupplyLoss, getTransshipmentFrom, sendTransshipmentTo, unitTransRevenueAnotherRetailer, anotherInventoryLevel):
        unmetDemand = max(thisRealizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + min(anotherInventoryLevel, getTransshipmentFrom) - min(self.inventoryLevel, sendTransshipmentTo)), 0)
        
        return unmetDemand
    
    
    
    
    def GetRealizedReward_online_forTest(self, realizedDemand, realizedSupplyLoss, anotherRetailer):
        # sampling q-learning
        
        realizedReward = ((self.unitRevenue * min(realizedDemand, self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + anotherRetailer.yAmount - self.yAmount)) 
                          - (self.unitOrderingCost * self.orderAmount)
                          - (self.unitHoldingCost * (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + anotherRetailer.yAmount - self.yAmount)) 
                          - (self.unitPenaltyforUnmetDemand * max(realizedDemand - (self.inventoryLevel + max(self.orderAmount - realizedSupplyLoss, 0) + anotherRetailer.yAmount - self.yAmount), 0))
                          - (self.unitTransshipmentCost * self.yAmount)
                          + (self.unitTransshipmentRevenue * self.yAmount)
                          - (anotherRetailer.unitTransshipmentRevenue * anotherRetailer.yAmount)) 
        
        return realizedReward
    
    
    
    