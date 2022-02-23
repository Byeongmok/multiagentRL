from Simulator.CumPi import CumPi
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Simulator.MultipleRegression import MultipleRegression
from sklearn.preprocessing import StandardScaler
from Population_GA.Population import Population
import copy
from tqdm import tqdm

class Sim():
    def __init__(self, type, stateLength1, stateLength2, actionLength1, actionLength2, xLength1, xLength2, oLength):
        self.learningRate = 1.0
        self.discountFactor = 0.9                                                        # Littman, 1994
        self.decayFactor = (10.0**(np.log(0.01) / (10.0**(6.0))))                        # Littman, 1994
        self.exploreFactor = 0.2                                                         # Littman, 1994
        

        self.a1_Index = 9999
        self.a2_Index = 9999
        self.x1_Index = 9999
        self.x2_Index = 9999
        self.x1hat_Index = 9999
        self.x2hat_Index = 9999
        self.o_Index = 9999
        self.o_prime_Index = 9999
        self.currentStateIndex1 = 9999
        self.currentStateIndex2 = 9999
        self.nextStateIndex1 = 9999
        self.nextStateIndex2 = 9999
        
        self.test_AvgLoss1 = []
        self.test_AvgLoss2 = []
        
        # initialize online learning model
        torch.manual_seed(1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model1 = MultipleRegression(6)
        self.model2 = MultipleRegression(6)
        
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)
        
        self.criterion1 = nn.MSELoss()
        self.optimizer1 = optim.Adam(self.model1.parameters(), lr=1e-4)
        
        self.criterion2 = nn.MSELoss()
        self.optimizer2 = optim.Adam(self.model2.parameters(), lr=1e-4)
        
        
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()

        
        self.x1_train = -9999
        self.x2_train = -9999
        
        self.learnBool = False
        
        
        self.scaler3Bool = False
        self.scaler4Bool = False
        
        self.beta1 = -9999
        self.beta2 = -9999
        
        if type == 2:   # centralized q-learning
            self.q_centralized = np.zeros((stateLength1, stateLength2, actionLength1, actionLength2, xLength1, xLength2))      # initialize to zeros
        
        if type == 3:   # centralized new q-learning
            self.q_centralized_new = np.zeros((stateLength1, stateLength2, actionLength1, actionLength2, oLength))      # initialize to zeros
            self.q_cnt = np.zeros((stateLength1, stateLength2, actionLength1, actionLength2, oLength))      # initialize to zeros
            
     
     
    def InitializePi_balancedGap_New(self, stateLength1, stateLength2, actionLength1, actionLength2, actionOLength):
        self.pi1 = np.ones(shape=[stateLength1, stateLength2, actionLength1])
        self.pi1 = self.pi1 / (actionLength1)   
        
        self.pi2 = np.ones(shape=[stateLength1, stateLength2, actionLength2])
        self.pi2 = self.pi2 / (actionLength2)   
        
        self.pio = np.ones(shape=[stateLength1, stateLength2, actionOLength])
        self.pio = self.pio / (actionOLength)   
        
        
    def InitializePi_balancedGap(self, stateLength1, stateLength2, action1Length, action2Length, actionOLength):
        self.pi = np.ones(shape=[stateLength1, stateLength2, action1Length, action2Length, actionOLength])
        self.pi = self.pi / (action1Length * action2Length * actionOLength)
        
        
        
    def InitializePi_decentralized(self, stateLength1, stateLength2, action1Length, action2Length):
        self.pi1 = np.ones(shape=[stateLength1, action1Length])
        self.pi1 = self.pi1 / (action1Length)

        self.pi2 = np.ones(shape=[stateLength2, action2Length])
        self.pi2 = self.pi2 / (action2Length)


    def InitializePi_centralized(self, stateLength1, stateLength2, action1Length, action2Length, x1Length, x2Length):
        self.pi = np.ones(shape=[stateLength1, stateLength2, action1Length, action2Length, x1Length, x2Length])
        self.pi = self.pi / (action1Length * action2Length * x1Length * x2Length)



    def InitializePi_centralized_New(self, stateLength1, stateLength2, action1Length, action2Length, actionOLength):
        self.pi = np.ones(shape=[stateLength1, stateLength2, action1Length, action2Length, actionOLength])
        self.pi = self.pi / (action1Length * action2Length * actionOLength)

    
    
    def GetIndexFromInverseCumulativePi_balancedGap_New(self, stateIndex1, stateIndex2, probability1, probability2, probability3, a1, a2, o):
        a1_Index = 9999
        a2_Index = 9999
        o_Index = 9999
        
        sum = 0.0
        cumPiList = []
        
        for i in range(len(a1.action)):
            sum += self.pi1[stateIndex1, stateIndex2, i]
            cumPiList.append(CumPi(i, -9999, -9999, sum, -9999, -9999))
        
        
        
        for i in range(len(cumPiList)):
            if i == 0:
                if probability1 < cumPiList[i].cumProbability:
                    a1_Index = cumPiList[i].a1_Index                    
                    break
                
            else:
                if cumPiList[i - 1].cumProbability <= probability1 and probability1 < cumPiList[i].cumProbability:
                    a1_Index = cumPiList[i].a1_Index
                    break
        

        
        sum = 0.0
        cumPiList = []
        
        for i in range(len(a2.action)):
            sum += self.pi2[stateIndex1, stateIndex2, i]
            cumPiList.append(CumPi(-9999, i, -9999, sum, -9999, -9999))
        
        
        
        for i in range(len(cumPiList)):
            if i == 0:
                if probability2 < cumPiList[i].cumProbability:
                    a2_Index = cumPiList[i].a2_Index                    
                    break
                
            else:
                if cumPiList[i - 1].cumProbability <= probability2 and probability2 < cumPiList[i].cumProbability:
                    a2_Index = cumPiList[i].a2_Index
                    break
        
        

        
        sum = 0.0
        cumPiList = []
        
        for i in range(len(o.action)):
            sum += self.pio[stateIndex1, stateIndex2, i]
            cumPiList.append(CumPi(-9999, -9999, i, sum, -9999, -9999))
        
        
        
        for i in range(len(cumPiList)):
            if i == 0:
                if probability3 < cumPiList[i].cumProbability:
                    o_Index = cumPiList[i].o_Index                    
                    break
                
            else:
                if cumPiList[i - 1].cumProbability <= probability3 and probability3 < cumPiList[i].cumProbability:
                    o_Index = cumPiList[i].o_Index
                    break
        
        
        return (a1_Index, a2_Index, o_Index)
    
    
    
    def GetIndexFromInverseCumulativePi_balancedGap_SARSA(self, stateIndex1, stateIndex2, probability, o):
        
        o_Index = 9999
        
        
        sum = 0.0
        cumPiList = []
        
        for i in range(len(o.action)):
            sum += self.pio[stateIndex1, stateIndex2, i]
            cumPiList.append(CumPi(-9999, -9999, i, sum, -9999, -9999))
        
        
        
        for i in range(len(cumPiList)):
            if i == 0:
                if probability < cumPiList[i].cumProbability:
                    o_Index = cumPiList[i].o_Index                    
                    break
                
            else:
                if cumPiList[i - 1].cumProbability <= probability and probability < cumPiList[i].cumProbability:
                    o_Index = cumPiList[i].o_Index
                    break
        
        
        return o_Index
    
    
    
    def GetIndexFromInverseCumulativePi_decentralized(self, stateIndex1, stateIndex2, probability1, probability2, a1, a2):
        
        sum = 0.0
        cumPiList1 = []
        
        for i in range(len(a1.action)):
            sum += self.pi1[stateIndex1, i]
            cumPiList1.append(CumPi(i, -9999, -9999, sum, -9999, -9999))
        
        
        for i in range(len(cumPiList1)):
            if i == 0:
                if probability1 < cumPiList1[i].cumProbability:
                    a1_Index = cumPiList1[i].a1_Index
                    break
                
            else:
                if cumPiList1[i - 1].cumProbability <= probability1 and probability1 < cumPiList1[i].cumProbability:
                    a1_Index = cumPiList1[i].a1_Index
                    break
         
                
        sum = 0.0
        cumPiList2 = []
        
        for i in range(len(a2.action)):
            sum += self.pi2[stateIndex2, i]
            cumPiList2.append(CumPi(-9999, i, -9999, sum, -9999, -9999))
        
        
        for i in range(len(cumPiList2)):
            if i == 0:
                if probability2 < cumPiList2[i].cumProbability:
                    a2_Index = cumPiList2[i].a2_Index
                    break
                
            else:
                if cumPiList2[i - 1].cumProbability <= probability2 and probability2 < cumPiList2[i].cumProbability:
                    a2_Index = cumPiList2[i].a2_Index
                    break
        
        return (a1_Index, a2_Index)
    
    
    
    def GetIndexFromInverseCumulativePi_centralized(self, stateIndex1, stateIndex2, probability, a1, a2, x1, x2):
        a1_Index = 9999
        a2_Index = 9999
        x1_Index = 9999
        x2_Index = 9999
        
        sum = 0.0
        cumPiList = []
        
        for i in range(len(a1.action)):
            for j in range(len(a2.action)):
                for k in range(len(x1.action)):
                    for l in range(len(x2.action)):
                        sum += self.pi[stateIndex1, stateIndex2, i, j, k, l]
                        cumPiList.append(CumPi(i, j, -9999, sum, k, l))
        
        
        for i in range(len(cumPiList)):
            if i == 0:
                if probability < cumPiList[i].cumProbability:
                    a1_Index = cumPiList[i].a1_Index
                    a2_Index = cumPiList[i].a2_Index
                    x1_Index = cumPiList[i].x1_Index
                    x2_Index = cumPiList[i].x2_Index
                    
                    break
                
            else:
                if cumPiList[i - 1].cumProbability <= probability and probability < cumPiList[i].cumProbability:
                    a1_Index = cumPiList[i].a1_Index
                    a2_Index = cumPiList[i].a2_Index
                    x1_Index = cumPiList[i].x1_Index
                    x2_Index = cumPiList[i].x2_Index
                    
                    break
        
        return (a1_Index, a2_Index, x1_Index, x2_Index)
    
    def GetIndexFromInverseCumulativePi_centralized_New(self, stateIndex1, stateIndex2, probability, a1, a2, o):
        a1_Index = 9999
        a2_Index = 9999
        o_Index = 9999
        
        sum = 0.0
        cumPiList = []
        
        for i in range(len(a1.action)):
            for j in range(len(a2.action)):
                for k in range(len(o.action)):
                    sum += self.pi[stateIndex1, stateIndex2, i, j, k]
                    cumPiList.append(CumPi(i, j, k, sum, -9999, -9999))
        
        
        for i in range(len(cumPiList)):
            if i == 0:
                if probability < cumPiList[i].cumProbability:
                    a1_Index = cumPiList[i].a1_Index
                    a2_Index = cumPiList[i].a2_Index
                    o_Index = cumPiList[i].o_Index
                    break
                
            else:
                if cumPiList[i - 1].cumProbability <= probability and probability < cumPiList[i].cumProbability:
                    a1_Index = cumPiList[i].a1_Index
                    a2_Index = cumPiList[i].a2_Index
                    o_Index = cumPiList[i].o_Index
                    break
        
        return (a1_Index, a2_Index, o_Index)
    
    
    
    def GetIndexFromInverseCumulativePi_balancedGap(self, stateIndex1, stateIndex2, probability, a1, a2, o):
        a1_Index = 9999
        a2_Index = 9999
        o_Index = 9999
        
        sum = 0.0
        cumPiList = []
        
        for i in range(len(a1.action)):
            for j in range(len(a2.action)):
                for k in range(len(o.action)):
                    sum += self.pi[stateIndex1, stateIndex2, i, j, k]
                    cumPiList.append(CumPi(i, j, k, sum, -9999, -9999))
        
        for i in range(len(cumPiList)):
            if i == 0:
                if probability < cumPiList[i].cumProbability:
                    a1_Index = cumPiList[i].a1_Index
                    a2_Index = cumPiList[i].a2_Index
                    o_Index = cumPiList[i].o_Index
                    
                    break
                
            else:
                if cumPiList[i - 1].cumProbability <= probability and probability < cumPiList[i].cumProbability:
                    a1_Index = cumPiList[i].a1_Index
                    a2_Index = cumPiList[i].a2_Index
                    o_Index = cumPiList[i].o_Index
                    
                    break
        
        return ((a1_Index, a2_Index), o_Index)
    
        
    
    def GetStateIndex(self, inventory1, inventory2, s):
        stateIndex1 = 9999
        stateIndex2 = 9999
        
        flag1 = False
        flag2 = False
        for i in range(len(s.state) - 1):
            if s.state[i][0][0] <= inventory1 and inventory1 < s.state[i + 1][0][0]:
                stateIndex1 = i
                flag1 = True
                break
        
        if(not flag1):
            if inventory1 >= s.state[len(s.state) - 1][0][0]:
                stateIndex1 = len(s.state) - 1
            else:
                stateIndex1 = 0
            
                
        for i in range(len(s.state) - 1):
            if s.state[0][i][1] <= inventory2 and inventory2 < s.state[0][i + 1][1]:
                stateIndex2 = i
                flag2 = True
                break
            
        if(not flag2):
            if inventory2 >= s.state[len(s.state) - 1][0][0]:
                stateIndex2 = len(s.state) - 1
            else:
                stateIndex2 = 0
        
        return (stateIndex1, stateIndex2)
    
    
    def GetMarginalPi(self, pi):
        
        
        marginalPi = pi.sum(axis=2)
        marginalPi = pi.sum(axis=2)
            
            
        return marginalPi
    

    def GetMaxDiscountedFutureQValue_balancedGap(self, retailer, a, o, nextStateIndex1, nextStateIndex2, marginalPi):
        # for balanced gap Q-learning
        
        q_valueList = []
        
        
        for i in range(len(a.action)):
            sum = 0.0
            for j in range(len(o.action)):
                sum += (retailer.q_balanced[nextStateIndex1, nextStateIndex2, i, j] * marginalPi[nextStateIndex1, nextStateIndex2, j])
                
            q_valueList.append(sum)
            
        return np.max(q_valueList)
    
    
    
    def GetMaxDiscountedFutureQValue_balancedGap_new(self, retailer, retailerIndex, a, o, nextStateIndex1, nextStateIndex2):
        # for balanced gap Q-learning
        
        q_valueList = []
        
        
        if retailerIndex == 1:
            
            
            for j in range(len(o.action)):
                sum = 0.0
                for i in range(len(a.action)):
                    sum += (retailer.q_balanced_new[nextStateIndex1, nextStateIndex2, i, j] * self.pi1[nextStateIndex1, nextStateIndex2, i])
                q_valueList.append(sum)
            return np.max(q_valueList)        
                
                
        if retailerIndex == 2:
            for j in range(len(o.action)):
                sum = 0.0
                for i in range(len(a.action)):
                    sum += (retailer.q_balanced_new[nextStateIndex1, nextStateIndex2, i, j] * self.pi2[nextStateIndex1, nextStateIndex2, i])
                q_valueList.append(sum)
            return np.max(q_valueList)        
            
        if retailerIndex == 0:
            
            q_valueList.append(retailer.q_balanced_new[nextStateIndex1, nextStateIndex2, :])
            
            
            return np.max(q_valueList)
    
    
    
    def GetMaxDiscountedFutureQValue_balancedGap_SARSA(self, retailer, retailerIndex, a, o_prime, nextStateIndex1, nextStateIndex2):
        
      
        if retailerIndex == 1:
            
            sum = 0.0
            for i in range(len(a.action)):
                sum += (retailer.q_balanced_new[nextStateIndex1, nextStateIndex2, i, o_prime] * self.pi1[nextStateIndex1, nextStateIndex2, i])
            
            return sum
            
            
                
        if retailerIndex == 2:
            
            sum = 0.0
            for i in range(len(a.action)):
                sum += (retailer.q_balanced_new[nextStateIndex1, nextStateIndex2, i, o_prime] * self.pi2[nextStateIndex1, nextStateIndex2, i])
            
            return sum        
            
            
        if retailerIndex == 0:
            
            return retailer.q_balanced_new[nextStateIndex1, nextStateIndex2, o_prime]
            
    
    
    def GetMaxDiscountedFutureQValue_decentralized(self, retailer, nextStateIndex):
        # for decentralized Q-learning
        
        return np.max(retailer.q_decentralized[nextStateIndex, :])
        
        
        
    def GetMaxDiscountedFutureQValue_centralized(self, nextStateIndex1, nextStateIndex2):
        # for centralized Q-learning
        
        return np.max(self.q_centralized[nextStateIndex1, nextStateIndex2, :, :, :, :])
    
    
    def GetMaxDiscountedFutureQValue_centralized_new(self, nextStateIndex1, nextStateIndex2):
        # for centralized Q-learning
        
        return np.max(self.q_centralized_new[nextStateIndex1, nextStateIndex2, :, :, :])
    
    
    def GetAverageAndStdDev_balancedGap(self, currentStateIndex1, currentStateIndex2, Q):
        average = np.mean(Q[currentStateIndex1, currentStateIndex2, :, :, :])
        stdDev = np.std(Q[currentStateIndex1, currentStateIndex2, :, :, :])
        
        return (average, stdDev)
    
    
    def GetAverageAndStdDev_balancedGap_new_retailer(self, currentStateIndex1, currentStateIndex2, Q):
        average = np.mean(Q[currentStateIndex1, currentStateIndex2, :, :])
        stdDev = np.std(Q[currentStateIndex1, currentStateIndex2, :, :])
        
        return (average, stdDev)
    
    def GetAverageAndStdDev_balancedGap_new_parent(self, currentStateIndex1, currentStateIndex2, Q):
        average = np.mean(Q[currentStateIndex1, currentStateIndex2, :])
        stdDev = np.std(Q[currentStateIndex1, currentStateIndex2, :])
        
        return (average, stdDev)
    
    
    def GetAverageAndStdDev_decentralized(self, currentStateIndex, Q):
        average = np.mean(Q[currentStateIndex, :])
        stdDev = np.std(Q[currentStateIndex, :])
        
        return (average, stdDev)
    
    
    def GetAverageAndStdDev_centralized(self, currentStateIndex1, currentStateIndex2):
        average = np.mean(self.q_centralized[currentStateIndex1, currentStateIndex2, :, :, :, :])
        stdDev = np.std(self.q_centralized[currentStateIndex1, currentStateIndex2, :, :, :, :])
        
        return (average, stdDev)
    
    def GetAverageAndStdDev_centralized_new(self, currentStateIndex1, currentStateIndex2):
        average = np.mean(self.q_centralized_new[currentStateIndex1, currentStateIndex2, :, :, :])
        stdDev = np.std(self.q_centralized_new[currentStateIndex1, currentStateIndex2, :, :, :])
        
        return (average, stdDev)


    
    
    def N(self, q, meanStdDev):
        return ((q - meanStdDev[0]) / (meanStdDev[1] + 1))
    

    def UpdatePi_balancedGap(self, parent, currentStateIndex1, currentStateIndex2, a1Length, a2Length, oLength):
        # for balanced gap Q-learning
        
        meanStdDev = self.GetAverageAndStdDev_balancedGap(currentStateIndex1, currentStateIndex2, parent.q_balanced)
        
        denominator = 0.0
        for i in range(a1Length):
            for j in range(a2Length):
                for k in range(oLength):
                    
                    denominator += np.exp(self.N(parent.q_balanced[currentStateIndex1, currentStateIndex2, i, j, k], meanStdDev))
                    
                    
        for i in range(a1Length):
            for j in range(a2Length):
                for k in range(oLength):
                    self.pi[currentStateIndex1, currentStateIndex2, i, j, k] = np.exp(self.N(parent.q_balanced[currentStateIndex1, currentStateIndex2, i, j, k], meanStdDev)) / denominator
    
    
    def UpdatePi_balancedGap_new(self, parent, retailer1, retailer2,  currentStateIndex1, currentStateIndex2, a1Length, a2Length, oLength):
        # for balanced gap Q-learning
        
        meanStdDev1 = self.GetAverageAndStdDev_balancedGap_new_retailer(currentStateIndex1, currentStateIndex2, retailer1.q_balanced_new)
        meanStdDev2 = self.GetAverageAndStdDev_balancedGap_new_retailer(currentStateIndex1, currentStateIndex2, retailer2.q_balanced_new)
        meanStdDevo = self.GetAverageAndStdDev_balancedGap_new_parent(currentStateIndex1, currentStateIndex2, parent.q_balanced_new)
        
        denominator = 0.0
        for i in range(a1Length):
            for j in range(oLength):
                denominator += np.exp(self.N(retailer1.q_balanced_new[currentStateIndex1, currentStateIndex2, i, j], meanStdDev1))
                    
        
        for i in range(a1Length):
            sum = 0.0           
            for j in range(oLength):
                sum += np.exp(self.N(retailer1.q_balanced_new[currentStateIndex1, currentStateIndex2, i, j], meanStdDev1))
                
            self.pi1[currentStateIndex1, currentStateIndex2, i] = sum / denominator                        
    
    
        denominator = 0.0
        for i in range(a2Length):
            for j in range(oLength):
                denominator += np.exp(self.N(retailer2.q_balanced_new[currentStateIndex1, currentStateIndex2, i, j], meanStdDev2))
                    
                    
        for i in range(a2Length):
            sum = 0.0
            for j in range(oLength):
                sum += np.exp(self.N(retailer2.q_balanced_new[currentStateIndex1, currentStateIndex2, i, j], meanStdDev2))                    
    
            self.pi2[currentStateIndex1, currentStateIndex2, i] = sum / denominator                        
        
        
        
        denominator = 0.0
        for i in range(oLength):
            
            denominator += np.exp(self.N(parent.q_balanced_new[currentStateIndex1, currentStateIndex2, i], meanStdDevo))
                    
                    
        for i in range(oLength):
                              
            self.pio[currentStateIndex1, currentStateIndex2, i] = np.exp(self.N(parent.q_balanced_new[currentStateIndex1, currentStateIndex2, i], meanStdDevo)) / denominator                    
    
    
    
    def UpdatePi_decentralized(self, retailer1, retailer2, currentStateIndex1, currentStateIndex2, a1Length, a2Length):
        # for decentralized Q-learning
        
        meanStdDev1 = self.GetAverageAndStdDev_decentralized(currentStateIndex1, retailer1.q_decentralized)
        meanStdDev2 = self.GetAverageAndStdDev_decentralized(currentStateIndex2, retailer2.q_decentralized)
        
        denominator = 0.0
        for i in range(a1Length):
            denominator += np.exp(self.N(retailer1.q_decentralized[currentStateIndex1, i], meanStdDev1))
                    
                    
        for i in range(a1Length):
            self.pi1[currentStateIndex1, i] = np.exp(self.N(retailer1.q_decentralized[currentStateIndex1, i], meanStdDev1)) / denominator
            

  
    
        denominator = 0.0
        for i in range(a2Length):
            denominator += np.exp(self.N(retailer2.q_decentralized[currentStateIndex2, i], meanStdDev2))
        
                  
                    
        for i in range(a2Length):
            self.pi2[currentStateIndex2, i] = np.exp(self.N(retailer2.q_decentralized[currentStateIndex2, i], meanStdDev2)) / denominator
            

        
    
    
    
    def UpdatePi_centralized(self, currentStateIndex1, currentStateIndex2, a1Length, a2Length, x1Length, x2Length):
        # for centralized Q-learning
        
        meanStdDev = self.GetAverageAndStdDev_centralized(currentStateIndex1, currentStateIndex2)
        
        denominator = 0.0
        for i in range(a1Length):
            for j in range(a2Length):
                for k in range(x1Length):
                    for l in range(x2Length):
                        denominator += np.exp(self.N(self.q_centralized[currentStateIndex1, currentStateIndex2, i, j, k, l], meanStdDev))
                    
                    
        for i in range(a1Length):
            for j in range(a2Length):
                for k in range(x1Length):
                    for l in range(x2Length):
                        self.pi[currentStateIndex1, currentStateIndex2, i, j, k, l] = np.exp(self.N(self.q_centralized[currentStateIndex1, currentStateIndex2, i, j, k, l], meanStdDev)) / denominator
                    
    
    def UpdatePi_centralized_new(self, currentStateIndex1, currentStateIndex2, a1Length, a2Length, oLength):
        # for centralized Q-learning
        
        meanStdDev = self.GetAverageAndStdDev_centralized_new(currentStateIndex1, currentStateIndex2)
        
        denominator = 0.0
        for i in range(a1Length):
            for j in range(a2Length):
                for k in range(oLength):
                    denominator += np.exp(self.N(self.q_centralized_new[currentStateIndex1, currentStateIndex2, i, j, k], meanStdDev))
                    
                    
        for i in range(a1Length):
            for j in range(a2Length):
                for k in range(oLength):
                    self.pi[currentStateIndex1, currentStateIndex2, i, j, k] = np.exp(self.N(self.q_centralized_new[currentStateIndex1, currentStateIndex2, i, j, k], meanStdDev)) / denominator
                    
    
    
    
    
    def GetRealizedReward_centralized(self, realizedDemand1, realizedDemand2, realizedSupplyLoss1, realizedSupplyLoss2, retailer1, retailer2):
        # centralized q-learning
        realizedReward = retailer1.GetRealizedReward_centralized(realizedDemand1, realizedSupplyLoss1, retailer2) + retailer2.GetRealizedReward_centralized(realizedDemand2, realizedSupplyLoss2, retailer1)
        
        
        return realizedReward
            
    
    
    def Run_balancedGap(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s,
            seedRndEGreedy, seedRndAction1, seedRndAction2, seedRndActionO, seedRndPi, initialInventoryLevel1, initialInventoryLevel2):
        # for balanced gap Q-learning
        
        rndEGreedyGenerator = random.Random(seedRndEGreedy)
        rndAction1Generator = random.Random(seedRndAction1)
        rndAction2Generator = random.Random(seedRndAction2)
        rndActionOGenerator = random.Random(seedRndActionO)
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel1
        retailer2.inventoryLevel = initialInventoryLevel2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndEGreedy = rndEGreedyGenerator.random()
            rndAction1 = rndAction1Generator.random()
            rndAction2 = rndAction2Generator.random()
            rndActionO = rndActionOGenerator.random()
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            if rndEGreedy <= self.exploreFactor:
                # a1
                for i in range(len(a1.action)):
                    if (((1.0 / len(a1.action)) * i) <= rndAction1) and (rndAction1 < ((1.0 / len(a1.action)) * (i + 1))):
                        self.a1_Index = i
                        break
                
                # a2
                for i in range(len(a2.action)):
                    if (((1.0 / len(a2.action)) * i) <= rndAction2) and (rndAction2 < ((1.0 / len(a2.action)) * (i + 1))):
                        self.a2_Index = i
                        break
                
                # o
                for i in range(len(o.action)):
                    if (((1.0 / len(o.action)) * i) <= rndActionO) and (rndActionO < ((1.0 / len(o.action)) * (i + 1))):
                        self.o_Index = i
                        break
                
            else:
                ((self.a1_Index, self.a2_Index), self.o_Index) = self.GetIndexFromInverseCumulativePi_balancedGap(currentStateIndex1, currentStateIndex2, rndPi, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R1(s, a1, o)
            currentReward1 = retailer1.GetRealizedReward_balancedGap(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            # get R2(s, a2, o)
            currentReward2 = retailer2.GetRealizedReward_balancedGap(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)                                                         
            
            # get next inventory level
            nextInventoryLevel1 = retailer1.GetInventoryUpdate_balancedGap(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)
            nextInventoryLevel2 = retailer2.GetInventoryUpdate_balancedGap(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            # find next state indices
            (nextStateIndex1, nextStateIndex2) = self.GetStateIndex(nextInventoryLevel1, nextInventoryLevel2, s)
            
            
            # get marginal probability pi(s, o) from pi(s, a1, a2, o)
            marginalPi_1 = self.GetMarginalPi(self.pi)
            
            # get marginal probability pi(s, o) from pi(s, a1, a2, o)
            marginalPi_2 = self.GetMarginalPi(self.pi)
            
            
            # update Q1(s, a1, o)
            retailer1.q_balanced[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index] = (((1.0 - self.learningRate) * retailer1.q_balanced[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index]) 
            + (self.learningRate * (currentReward1 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap(retailer1, a1, o, nextStateIndex1, nextStateIndex2, marginalPi_1)))))
            
            
            # update Q2(s, a2, o)
            retailer2.q_balanced[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index] = (((1.0 - self.learningRate) * retailer2.q_balanced[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index]) 
            + (self.learningRate * (currentReward2 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap(retailer2, a2, o, nextStateIndex1, nextStateIndex2, marginalPi_2)))))
            
            
            # update Qo(s, a1, a2, o)
            parent.q_balanced[currentStateIndex1, currentStateIndex2, self.a1_Index, self.a2_Index, self.o_Index] = retailer1.q_balanced[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index] + retailer2.q_balanced[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index]
            
            
            # update pi(s, a1, a2, o) by using Softmax rule
            self.UpdatePi_balancedGap(parent, currentStateIndex1, currentStateIndex2, len(a1.action), len(a2.action), len(o.action))


            # learning rate update
            self.learningRate *= self.decayFactor
            
            
            # inventory update
            retailer1.InventoryUpdate_balancedGap(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_balancedGap(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            # t <- t+1
            t += 1
    
    def Test_balancedGap(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s, seedRndPi, initialInventoryLevel_1, initialInventoryLevel_2):
        # for balanced gap New Q-learning
        
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel_1
        retailer2.inventoryLevel = initialInventoryLevel_2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            
            ((self.a1_Index, self.a2_Index), self.o_Index) = self.GetIndexFromInverseCumulativePi_balancedGap(currentStateIndex1, currentStateIndex2, rndPi, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R1(s, a1, o)
            currentReward1 = retailer1.GetRealizedReward_balancedGap(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            # get R2(s, a2, o)
            currentReward2 = retailer2.GetRealizedReward_balancedGap(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)                                                             
            
            
            # get UnmetDemand
            unmetDemand1 = retailer1.GetUnmetDemand_balancedGap(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.inventoryLevel)
            
            unmetDemand2 = retailer2.GetUnmetDemand_balancedGap(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            
            # get Amount Trans To
            amountTransTo1 = min(retailer1.inventoryLevel, o.action[self.o_Index][0])
            amountTransTo2 = min(retailer2.inventoryLevel, o.action[self.o_Index][1])
            
            
            
            # save results for printing out
            retailer1.inventoryLevelBeginningList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelBeginningList.append(retailer2.inventoryLevel)
            
            
            # inventory update
            retailer1.InventoryUpdate_balancedGap(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_balancedGap(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            
            # save results for printing out
            retailer1.inventoryLevelRemainingList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelRemainingList.append(retailer2.inventoryLevel)
            
            retailer1.realizedRewardList.append(currentReward1)
            retailer2.realizedRewardList.append(currentReward2)
            
            retailer1.orderAmountList.append(retailer1.orderAmount)
            retailer2.orderAmountList.append(retailer2.orderAmount)
            
            parent.transshipmentAmountList.append(parent.transshipmentAmount)
            
            
            retailer1.unmetDemandList.append(unmetDemand1)
            retailer2.unmetDemandList.append(unmetDemand2)
            
            retailer1.amountTransToList.append(amountTransTo1)
            retailer2.amountTransToList.append(amountTransTo2)
            
            
            # t <- t+1
            t += 1
    
    
    
    def Run_balancedGap_SARSA(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s,
            seedRndEGreedy, seedRndE2Greedy, seedRndAction1, seedRndAction2, seedRndActionO, seedRndPi, initialInventoryLevel1, initialInventoryLevel2):
        # for balanced gap New Q-learning
        
        rndEGreedyGenerator = random.Random(seedRndEGreedy)
        rndEGreedyGenerator2 = random.Random(seedRndE2Greedy)
        rndAction1Generator = random.Random(seedRndAction1)
        rndAction2Generator = random.Random(seedRndAction2)
        rndActionOGenerator = random.Random(seedRndActionO)
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel1
        retailer2.inventoryLevel = initialInventoryLevel2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndEGreedy = rndEGreedyGenerator.random()
            rndEGreedy2 = rndEGreedyGenerator2.random()
            rndAction1 = rndAction1Generator.random()
            rndAction2 = rndAction2Generator.random()
            rndActionO = rndActionOGenerator.random()
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            if rndEGreedy <= self.exploreFactor:
                # a1
                for i in range(len(a1.action)):
                    if (((1.0 / len(a1.action)) * i) <= rndAction1) and (rndAction1 < ((1.0 / len(a1.action)) * (i + 1))):
                        self.a1_Index = i
                        break
                
                # a2
                for i in range(len(a2.action)):
                    if (((1.0 / len(a2.action)) * i) <= rndAction2) and (rndAction2 < ((1.0 / len(a2.action)) * (i + 1))):
                        self.a2_Index = i
                        break
                
                # o
                for i in range(len(o.action)):
                    if (((1.0 / len(o.action)) * i) <= rndActionO) and (rndActionO < ((1.0 / len(o.action)) * (i + 1))):
                        self.o_Index = i
                        break
                
            else:
                rndPi2 = rndPiGenerator.random()
                rndPi3 = rndPiGenerator.random()
                (self.a1_Index, self.a2_Index, self.o_Index) = self.GetIndexFromInverseCumulativePi_balancedGap_New(currentStateIndex1, currentStateIndex2, rndPi, rndPi2, rndPi3, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R1(s, a1, o)
            currentReward1 = retailer1.GetRealizedReward_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            # get R2(s, a2, o)
            currentReward2 = retailer2.GetRealizedReward_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)                                                         
            
            
            
            
            # get next inventory level
            nextInventoryLevel1 = retailer1.GetInventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)
            nextInventoryLevel2 = retailer2.GetInventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            # find next state indices
            (nextStateIndex1, nextStateIndex2) = self.GetStateIndex(nextInventoryLevel1, nextInventoryLevel2, s)
            
            
            # find o'(t+1)
            if rndEGreedy2 <= self.exploreFactor:
                
                rndActionO = rndActionOGenerator.random()
                
                
                for i in range(len(o.action)):
                    if (((1.0 / len(o.action)) * i) <= rndActionO) and (rndActionO < ((1.0 / len(o.action)) * (i + 1))):
                        self.o_prime_Index = i
                        break
                
            else:
                rndPi = rndPiGenerator.random()
                self.o_prime_Index = self.GetIndexFromInverseCumulativePi_balancedGap_SARSA(nextStateIndex1, nextStateIndex2, rndPi, o)
            
            
            
            # update Q1(s, a1, o)
            retailer1.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index] = (((1.0 - self.learningRate) * retailer1.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index]) 
            + (self.learningRate * (currentReward1 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_SARSA(retailer1, 1, a1, self.o_prime_Index, nextStateIndex1, nextStateIndex2)))))
            
            
            # update Q2(s, a2, o)
            retailer2.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index] = (((1.0 - self.learningRate) * retailer2.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index]) 
            + (self.learningRate * (currentReward2 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_SARSA(retailer2, 2, a2, self.o_prime_Index, nextStateIndex1, nextStateIndex2)))))
            
            
            # update Qo(s, o)
            parent.q_balanced_new[currentStateIndex1, currentStateIndex2, self.o_Index] = (((1.0 - self.learningRate) * parent.q_balanced_new[currentStateIndex1, currentStateIndex2, self.o_Index]) 
            + (self.learningRate * ((currentReward1 + currentReward2) + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_SARSA(parent, 0, o, self.o_prime_Index, nextStateIndex1, nextStateIndex2)))))
            
                                                                                                                     
            
            
            # update pi(s, a1, a2, o) by using Softmax rule
            self.UpdatePi_balancedGap_new(parent, retailer1, retailer2, currentStateIndex1, currentStateIndex2, len(a1.action), len(a2.action), len(o.action))


            # learning rate update
            self.learningRate *= self.decayFactor
            
            
            # inventory update
            retailer1.InventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            # t <- t+1
            t += 1
    
    
    
    
    def Run_balancedGap_New(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s,
            seedRndEGreedy, seedRndAction1, seedRndAction2, seedRndActionO, seedRndPi, initialInventoryLevel1, initialInventoryLevel2, riskSensitivity1, riskSensitivity2):
        # for balanced gap New Q-learning
        
        rndEGreedyGenerator = random.Random(seedRndEGreedy)
        rndAction1Generator = random.Random(seedRndAction1)
        rndAction2Generator = random.Random(seedRndAction2)
        rndActionOGenerator = random.Random(seedRndActionO)
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel1
        retailer2.inventoryLevel = initialInventoryLevel2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndEGreedy = rndEGreedyGenerator.random()
            rndAction1 = rndAction1Generator.random()
            rndAction2 = rndAction2Generator.random()
            rndActionO = rndActionOGenerator.random()
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel * riskSensitivity1, retailer2.inventoryLevel * riskSensitivity2, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            if rndEGreedy <= self.exploreFactor:
                # a1
                for i in range(len(a1.action)):
                    if (((1.0 / len(a1.action)) * i) <= rndAction1) and (rndAction1 < ((1.0 / len(a1.action)) * (i + 1))):
                        self.a1_Index = i
                        break
                
                # a2
                for i in range(len(a2.action)):
                    if (((1.0 / len(a2.action)) * i) <= rndAction2) and (rndAction2 < ((1.0 / len(a2.action)) * (i + 1))):
                        self.a2_Index = i
                        break
                
                # o
                for i in range(len(o.action)):
                    if (((1.0 / len(o.action)) * i) <= rndActionO) and (rndActionO < ((1.0 / len(o.action)) * (i + 1))):
                        self.o_Index = i
                        break
                
            else:
                rndPi2 = rndPiGenerator.random()
                rndPi3 = rndPiGenerator.random()
                (self.a1_Index, self.a2_Index, self.o_Index) = self.GetIndexFromInverseCumulativePi_balancedGap_New(currentStateIndex1, currentStateIndex2, rndPi, rndPi2, rndPi3, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R1(s, a1, o)
            currentReward1 = retailer1.GetRealizedReward_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            # get R2(s, a2, o)
            currentReward2 = retailer2.GetRealizedReward_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)                                                         
            
            
            
            
            # get next inventory level
            nextInventoryLevel1 = retailer1.GetInventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)
            nextInventoryLevel2 = retailer2.GetInventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            # find next state indices
            (nextStateIndex1, nextStateIndex2) = self.GetStateIndex(nextInventoryLevel1 * riskSensitivity1, nextInventoryLevel2 * riskSensitivity2, s)
            
            
            # update Q1(s, a1, o)
            retailer1.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index] = (((1.0 - self.learningRate) * retailer1.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index]) 
            + (self.learningRate * (currentReward1 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_new(retailer1, 1, a1, o, nextStateIndex1, nextStateIndex2)))))
            
            
            # update Q2(s, a2, o)
            retailer2.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index] = (((1.0 - self.learningRate) * retailer2.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index]) 
            + (self.learningRate * (currentReward2 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_new(retailer2, 2, a2, o, nextStateIndex1, nextStateIndex2)))))
            
            
            # update Qo(s, o)
            parent.q_balanced_new[currentStateIndex1, currentStateIndex2, self.o_Index] = (((1.0 - self.learningRate) * parent.q_balanced_new[currentStateIndex1, currentStateIndex2, self.o_Index]) 
            + (self.learningRate * ((currentReward1 + currentReward2) + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_new(parent, 0, o, o, nextStateIndex1, nextStateIndex2)))))
            
            
            # Count visited (state, action) pair
            retailer1.q_cnt[currentStateIndex1, currentStateIndex2, self.a1_Index] = retailer1.q_cnt[currentStateIndex1, currentStateIndex2, self.a1_Index] + 1                                                                                                     
            retailer2.q_cnt[currentStateIndex1, currentStateIndex2, self.a2_Index] = retailer2.q_cnt[currentStateIndex1, currentStateIndex2, self.a2_Index] + 1
            parent.q_cnt[currentStateIndex1, currentStateIndex2, self.o_Index] = parent.q_cnt[currentStateIndex1, currentStateIndex2, self.o_Index] + 1
            
            
            # update pi(s, a1, a2, o) by using Softmax rule
            self.UpdatePi_balancedGap_new(parent, retailer1, retailer2, currentStateIndex1, currentStateIndex2, len(a1.action), len(a2.action), len(o.action))


            # learning rate update
            self.learningRate *= self.decayFactor
            
            
            # inventory update
            retailer1.InventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            # t <- t+1
            t += 1
    
    
    
    
    def Run_balancedGap_New_Online(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s,
            seedRndEGreedy, seedRndAction1, seedRndAction2, seedRndActionO, seedRndPi, initialInventoryLevel1, initialInventoryLevel2, riskSensitivity1, riskSensitivity2):
        # for balanced gap New Q-learning
        
        rndEGreedyGenerator = random.Random(seedRndEGreedy)
        rndAction1Generator = random.Random(seedRndAction1)
        rndAction2Generator = random.Random(seedRndAction2)
        rndActionOGenerator = random.Random(seedRndActionO)
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel1
        retailer2.inventoryLevel = initialInventoryLevel2
        
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndEGreedy = rndEGreedyGenerator.random()
            rndAction1 = rndAction1Generator.random()
            rndAction2 = rndAction2Generator.random()
            rndActionO = rndActionOGenerator.random()
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel * riskSensitivity1, retailer2.inventoryLevel * riskSensitivity2, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            if rndEGreedy <= self.exploreFactor:
                # a1
                for i in range(len(a1.action)):
                    if (((1.0 / len(a1.action)) * i) <= rndAction1) and (rndAction1 < ((1.0 / len(a1.action)) * (i + 1))):
                        self.a1_Index = i
                        break
                
                # a2
                for i in range(len(a2.action)):
                    if (((1.0 / len(a2.action)) * i) <= rndAction2) and (rndAction2 < ((1.0 / len(a2.action)) * (i + 1))):
                        self.a2_Index = i
                        break
                
                # o
                for i in range(len(o.action)):
                    if (((1.0 / len(o.action)) * i) <= rndActionO) and (rndActionO < ((1.0 / len(o.action)) * (i + 1))):
                        self.o_Index = i
                        break
                
            else:
                rndPi2 = rndPiGenerator.random()
                rndPi3 = rndPiGenerator.random()
                (self.a1_Index, self.a2_Index, self.o_Index) = self.GetIndexFromInverseCumulativePi_balancedGap_New(currentStateIndex1, currentStateIndex2, rndPi, rndPi2, rndPi3, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R1(s, a1, o)
            currentReward1, currentUnmetDemand1 = retailer1.GetRealizedReward_balancedGap_new_online(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            # get R2(s, a2, o)
            currentReward2, currentUnmetDemand2 = retailer2.GetRealizedReward_balancedGap_new_online(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)                                                         
            
            
            
            
            

            # target data for training
            # 완전 처음에는 training 시킬 데이터가 없음
            if self.learnBool == True:
                y1_target = [currentUnmetDemand1]
                # y1_target = self.scaler3.fit_transform(np.array(y1_target).reshape(-1, 1))
                y1_target = torch.Tensor(y1_target)
                y1_target = y1_target.to(self.device)
                # self.scaler3Bool = True
                
                y2_target = [currentUnmetDemand2]
                # y2_target = self.scaler4.fit_transform(np.array(y2_target).reshape(-1, 1))
                y2_target = torch.Tensor(y2_target)
                y2_target = y2_target.to(self.device)
                # self.scaler4Bool = True
            
            
                # training by using previous period features and current period target
                # The model predicts the current period target by using previous period features
                
                # self.x1_train: previous period features
                # y1_target = current period target
                
                # x1_train = self.scaler1.fit_transform(np.array(self.x1_train).reshape(-1, 1))
                # x1_train = torch.tensor(x1_train.reshape(1, -1).tolist())
                x1_train = torch.Tensor(self.x1_train)
                x1_train = x1_train.to(self.device)
                
                
                # x2_train = self.scaler2.fit_transform(np.array(self.x2_train).reshape(-1, 1))
                # x2_train = torch.tensor(x2_train.reshape(1, -1).tolist())
                x2_train = torch.Tensor(self.x2_train)
                x2_train = x2_train.to(self.device)
                
                
                nb_epochs = 1
                self.model1.train()
                for epoch in range(nb_epochs):
                    self.optimizer1.zero_grad()
                    
                    output = self.model1(x1_train)
                    loss = self.criterion1(output, y1_target)
                    loss.backward()
                    self.optimizer1.step()
                    
                    
                self.model2.train()
                for epoch in range(nb_epochs):
                    self.optimizer2.zero_grad()
                    
                    output = self.model2(x2_train)
                    loss = self.criterion1(output, y2_target)
                    loss.backward()
                    self.optimizer2.step()
            
            
            
            
            
            # data for training at next period
            self.x1_train = [demand1.realizedArray[t], supplyLoss1.realizedArray[t], retailer1.inventoryLevel, retailer1.orderAmount, parent.transshipmentAmount[0], parent.transshipmentAmount[1]]
            
            # if self.scaler3Bool == True:
                # x1_train = self.scaler1.transform(np.array(self.x1_train).reshape(-1, 1))
                # x1_train = torch.tensor(x1_train.reshape(1, -1).tolist())
            x1_train = torch.Tensor(self.x1_train)
            x1_train = x1_train.to(self.device)
   
            
            
            self.x2_train = [demand2.realizedArray[t], supplyLoss2.realizedArray[t], retailer2.inventoryLevel, retailer2.orderAmount, parent.transshipmentAmount[1], parent.transshipmentAmount[0]]
            
            # if self.scaler4Bool == True:
                # x2_train = self.scaler2.transform(np.array(self.x2_train).reshape(-1, 1))
                # x2_train = torch.tensor(x2_train.reshape(1, -1).tolist())
            x2_train = torch.Tensor(self.x2_train)
            x2_train = x2_train.to(self.device)
 
            
            
            
            
            # predict
            # 완전 처음에는 예측할 거도 없음 (훈련을 안했으니까)
            if self.learnBool == True:
                self.model1.eval()
                with torch.no_grad():
                    estimatedUnmedDemand1 = self.model1(x1_train)

                    # if self.scaler3Bool == True:
                        # estimatedUnmedDemand1 = round(self.scaler3.inverse_transform(estimatedUnmedDemand1.detach().cpu().numpy().reshape(-1, 1))[0][0])
                    estimatedUnmedDemand1 = round(estimatedUnmedDemand1.detach().cpu().numpy()[0])
          
                    
                    
                    
                self.model2.eval()
                with torch.no_grad():
                    estimatedUnmedDemand2 = self.model2(x2_train)

                    # if self.scaler4Bool == True:
                        # estimatedUnmedDemand2 = round(self.scaler4.inverse_transform(estimatedUnmedDemand2.detach().cpu().numpy().reshape(-1, 1))[0][0])
                    estimatedUnmedDemand2 = round(estimatedUnmedDemand2.detach().cpu().numpy()[0])
                    
                    
                    
            # Set beta based on estimatedUnmetDemand
            if self.learnBool == False:
                
                self.beta1 = 1.0
                self.beta2 = 1.0
                
                self.learnBool = True
            else:
                if estimatedUnmedDemand1 == 0:
                    self.beta1 = 1.0
                elif estimatedUnmedDemand1 > 0 and  estimatedUnmedDemand1 <= 10:
                    self.beta1 = 1.1
                elif estimatedUnmedDemand1 > 10 and estimatedUnmedDemand1 <= 20:
                    self.beta1 = 1.2
                elif estimatedUnmedDemand1 > 20 and estimatedUnmedDemand1 <= 30:
                    self.beta1 = 1.3
                elif estimatedUnmedDemand1 > 30 and estimatedUnmedDemand1 <= 40:
                    self.beta1 = 1.4
                elif estimatedUnmedDemand1 > 40 and estimatedUnmedDemand1 <= 50:
                    self.beta1 = 1.5
                elif estimatedUnmedDemand1 > 50:
                    self.beta1 = 1.6
                else:
                    self.beta1 = 1.0
                
                
                if estimatedUnmedDemand2 == 0:
                    self.beta2 = 1.0
                elif estimatedUnmedDemand2 > 0 and  estimatedUnmedDemand2 <= 10:
                    self.beta2 = 1.1
                elif estimatedUnmedDemand2 > 10 and estimatedUnmedDemand2 <= 20:
                    self.beta2 = 1.2
                elif estimatedUnmedDemand2 > 20 and estimatedUnmedDemand2 <= 30:
                    self.beta2 = 1.3
                elif estimatedUnmedDemand2 > 30 and estimatedUnmedDemand2 <= 40:
                    self.beta2 = 1.4
                elif estimatedUnmedDemand2 > 40 and estimatedUnmedDemand2 <= 50:
                    self.beta2 = 1.5
                elif estimatedUnmedDemand2 > 50:
                    self.beta2 = 1.6
                else:
                    self.beta2 = 1.0
                
                
            
            
            
            # get R1(s, a1, o)
            currentReward1Tilt = retailer1.GetRealizedReward_balancedGap_new_online_Tilt(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel, self.beta1)
            
            # get R2(s, a2, o)
            currentReward2Tilt = retailer2.GetRealizedReward_balancedGap_new_online_Tilt(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel, self.beta2)   
            
            
            # get next inventory level
            nextInventoryLevel1 = retailer1.GetInventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)
            nextInventoryLevel2 = retailer2.GetInventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            # find next state indices
            (nextStateIndex1, nextStateIndex2) = self.GetStateIndex(nextInventoryLevel1 * riskSensitivity1, nextInventoryLevel2 * riskSensitivity2, s)
            
            
            # update Q1(s, a1, o)
            retailer1.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index] = (((1.0 - self.learningRate) * retailer1.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a1_Index, self.o_Index]) 
            + (self.learningRate * (currentReward1 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_new(retailer1, 1, a1, o, nextStateIndex1, nextStateIndex2)))))
            
            
            # update Q2(s, a2, o)
            retailer2.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index] = (((1.0 - self.learningRate) * retailer2.q_balanced_new[currentStateIndex1, currentStateIndex2, self.a2_Index, self.o_Index]) 
            + (self.learningRate * (currentReward2 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_new(retailer2, 2, a2, o, nextStateIndex1, nextStateIndex2)))))
            
            
            # update Qo(s, o)
            parent.q_balanced_new[currentStateIndex1, currentStateIndex2, self.o_Index] = (((1.0 - self.learningRate) * parent.q_balanced_new[currentStateIndex1, currentStateIndex2, self.o_Index]) 
            + (self.learningRate * (np.abs(currentReward1Tilt - currentReward2Tilt) + (self.discountFactor * self.GetMaxDiscountedFutureQValue_balancedGap_new(parent, 0, o, o, nextStateIndex1, nextStateIndex2)))))
            
            
                                                                                                                     
            
            
            # update pi(s, a1, a2, o) by using Softmax rule
            self.UpdatePi_balancedGap_new(parent, retailer1, retailer2, currentStateIndex1, currentStateIndex2, len(a1.action), len(a2.action), len(o.action))


            # learning rate update
            self.learningRate *= self.decayFactor
            
            
            # inventory update
            retailer1.InventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            # t <- t+1
            t += 1
    
    
            
            
    def Test_balancedGap_SARSA(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s, seedRndPi, initialInventoryLevel_1, initialInventoryLevel_2):
        
        
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel_1
        retailer2.inventoryLevel = initialInventoryLevel_2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            rndPi2 = rndPiGenerator.random()
            rndPi3 = rndPiGenerator.random()
            (self.a1_Index, self.a2_Index, self.o_Index) = self.GetIndexFromInverseCumulativePi_balancedGap_New(currentStateIndex1, currentStateIndex2, rndPi, rndPi2, rndPi3, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R1(s, a1, o)
            currentReward1 = retailer1.GetRealizedReward_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            # get R2(s, a2, o)
            currentReward2 = retailer2.GetRealizedReward_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)                                                         
            
            
            # save results for printing out
            retailer1.inventoryLevelBeginningList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelBeginningList.append(retailer2.inventoryLevel)
            
            
            # inventory update
            retailer1.InventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            
            # save results for printing out
            retailer1.inventoryLevelRemainingList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelRemainingList.append(retailer2.inventoryLevel)
            
            retailer1.realizedRewardList.append(currentReward1)
            retailer2.realizedRewardList.append(currentReward2)
            
            retailer1.orderAmountList.append(retailer1.orderAmount)
            retailer2.orderAmountList.append(retailer2.orderAmount)
            
            parent.transshipmentAmountList.append(parent.transshipmentAmount)
            
            
            # t <- t+1
            t += 1
     
    
    
    def Test_balancedGap_New(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s, seedRndPi, initialInventoryLevel_1, initialInventoryLevel_2):
        # for balanced gap New Q-learning
        
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel_1
        retailer2.inventoryLevel = initialInventoryLevel_2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            rndPi2 = rndPiGenerator.random()
            rndPi3 = rndPiGenerator.random()
            (self.a1_Index, self.a2_Index, self.o_Index) = self.GetIndexFromInverseCumulativePi_balancedGap_New(currentStateIndex1, currentStateIndex2, rndPi, rndPi2, rndPi3, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R1(s, a1, o)
            currentReward1 = retailer1.GetRealizedReward_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            # get R2(s, a2, o)
            currentReward2 = retailer2.GetRealizedReward_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)                                                         
            
            # get UnmetDemand
            unmetDemand1 = retailer1.GetUnmetDemand_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            unmetDemand2 = retailer2.GetUnmetDemand_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)           
            
            # get Amount Trans To
            amountTransTo1 = min(retailer1.inventoryLevel, o.action[self.o_Index][0])
            amountTransTo2 = min(retailer2.inventoryLevel, o.action[self.o_Index][1])
            
            
            
            # save results for printing out
            retailer1.inventoryLevelBeginningList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelBeginningList.append(retailer2.inventoryLevel)
            
            
            # inventory update
            retailer1.InventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            
            # save results for printing out
            retailer1.inventoryLevelRemainingList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelRemainingList.append(retailer2.inventoryLevel)
            
            retailer1.realizedRewardList.append(currentReward1)
            retailer2.realizedRewardList.append(currentReward2)
            
            retailer1.orderAmountList.append(retailer1.orderAmount)
            retailer2.orderAmountList.append(retailer2.orderAmount)
            
            parent.transshipmentAmountList.append(parent.transshipmentAmount)
            
            retailer1.unmetDemandList.append(unmetDemand1)
            retailer2.unmetDemandList.append(unmetDemand2)
            
            retailer1.amountTransToList.append(amountTransTo1)
            retailer2.amountTransToList.append(amountTransTo2)
            
            # t <- t+1
            t += 1
    
    
    def Test_balancedGap_New_Online(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s, seedRndPi, initialInventoryLevel_1, initialInventoryLevel_2):
        # for balanced gap New Q-learning online
        
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel_1
        retailer2.inventoryLevel = initialInventoryLevel_2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            rndPi2 = rndPiGenerator.random()
            rndPi3 = rndPiGenerator.random()
            (self.a1_Index, self.a2_Index, self.o_Index) = self.GetIndexFromInverseCumulativePi_balancedGap_New(currentStateIndex1, currentStateIndex2, rndPi, rndPi2, rndPi3, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R1(s, a1, o)
            currentReward1, currentUnmetDemand1 = retailer1.GetRealizedReward_balancedGap_new_online(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            # get R2(s, a2, o)
            currentReward2, currentUnmetDemand2 = retailer2.GetRealizedReward_balancedGap_new_online(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)                                                         
            
            
            # save results for printing out
            retailer1.inventoryLevelBeginningList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelBeginningList.append(retailer2.inventoryLevel)
            
            
            # inventory update
            retailer1.InventoryUpdate_balancedGap_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_balancedGap_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            
            # save results for printing out
            retailer1.inventoryLevelRemainingList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelRemainingList.append(retailer2.inventoryLevel)
            
            retailer1.realizedRewardList.append(currentReward1)
            retailer2.realizedRewardList.append(currentReward2)
            
            retailer1.orderAmountList.append(retailer1.orderAmount)
            retailer2.orderAmountList.append(retailer2.orderAmount)
            
            parent.transshipmentAmountList.append(parent.transshipmentAmount)
            
            
            # t <- t+1
            t += 1
    
    
    def Run_decentralized(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, s,
            seedRndEGreedy, seedRndAction1, seedRndAction2, seedRndPi, initialInventoryLevel1, initialInventoryLevel2, riskSensitivity1, riskSensitivity2):
        # for decentralized Q-learning (Trans X)
        
        rndEGreedyGenerator = random.Random(seedRndEGreedy)
        rndAction1Generator = random.Random(seedRndAction1)
        rndAction2Generator = random.Random(seedRndAction2)
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel1
        retailer2.inventoryLevel = initialInventoryLevel2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndEGreedy = rndEGreedyGenerator.random()
            rndAction1 = rndAction1Generator.random()
            rndAction2 = rndAction2Generator.random()
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel * riskSensitivity1, retailer2.inventoryLevel * riskSensitivity2, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            if rndEGreedy <= self.exploreFactor:
                # a1
                for i in range(len(a1.action)):
                    if (((1.0 / len(a1.action)) * i) <= rndAction1) and (rndAction1 < ((1.0 / len(a1.action)) * (i + 1))):
                        self.a1_Index = i
                        break
                
                # a2
                for i in range(len(a2.action)):
                    if (((1.0 / len(a2.action)) * i) <= rndAction2) and (rndAction2 < ((1.0 / len(a2.action)) * (i + 1))):
                        self.a2_Index = i
                        break
                
                
            else:
                rndPi2 = rndPiGenerator.random()
                (self.a1_Index, self.a2_Index) = self.GetIndexFromInverseCumulativePi_decentralized(currentStateIndex1, currentStateIndex2, rndPi, rndPi2, a1, a2)
                
                
                
                 
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            
            # get R1(s, a1)
            currentReward1 = retailer1.GetRealizedReward_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])
            
            # get R2(s, a2)
            currentReward2 = retailer2.GetRealizedReward_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])                                                         
            
            
            # get next inventory level
            nextInventoryLevel1 = retailer1.GetInventoryUpdate_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])
            nextInventoryLevel2 = retailer2.GetInventoryUpdate_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])
            
            # find next state indices
            (nextStateIndex1, nextStateIndex2) = self.GetStateIndex(nextInventoryLevel1 * riskSensitivity1, nextInventoryLevel2 * riskSensitivity2, s)
            
  
            
            # update Q1(s, a1)
            retailer1.q_decentralized[currentStateIndex1, self.a1_Index] = (((1.0 - self.learningRate) * retailer1.q_decentralized[currentStateIndex1, self.a1_Index]) 
            + (self.learningRate * (currentReward1 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_decentralized(retailer1, nextStateIndex1)))))
            
            
            # update Q2(s, a2)
            retailer2.q_decentralized[currentStateIndex2, self.a2_Index] = (((1.0 - self.learningRate) * retailer2.q_decentralized[currentStateIndex2, self.a2_Index]) 
            + (self.learningRate * (currentReward2 + (self.discountFactor * self.GetMaxDiscountedFutureQValue_decentralized(retailer2, nextStateIndex2)))))
            
            
            # count the visited (state, action) pair
            retailer1.q_cnt[currentStateIndex1, self.a1_Index] = retailer1.q_cnt[currentStateIndex1, self.a1_Index] + 1
            retailer2.q_cnt[currentStateIndex2, self.a2_Index] = retailer2.q_cnt[currentStateIndex2, self.a2_Index] + 1
            
            
            # update pi(s1, a1) by using Softmax rule
            self.UpdatePi_decentralized(retailer1, retailer2, currentStateIndex1, currentStateIndex2, len(a1.action), len(a2.action))
            


            # learning rate update
            self.learningRate *= self.decayFactor
            
            
            # inventory update
            retailer1.InventoryUpdate_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])           
            retailer2.InventoryUpdate_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])
            
            
            # t <- t+1
            t += 1
            
            
    def Test_decentralized(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, a1, a2, s, seedRndPi, initialInventoryLevel_1, initialInventoryLevel_2):
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel_1
        retailer2.inventoryLevel = initialInventoryLevel_2
        
        t = 0
        while(t < maxT):
            # generate random number
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            rndPi2 = rndPiGenerator.random()
            (self.a1_Index, self.a2_Index) = self.GetIndexFromInverseCumulativePi_decentralized(currentStateIndex1, currentStateIndex2, rndPi, rndPi2, a1, a2)
            
            
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            
            # get R1(s, a1)
            currentReward1 = retailer1.GetRealizedReward_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])
            
            # get R2(s, a2)
            currentReward2 = retailer2.GetRealizedReward_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])                                                            
            
            
            # get UnmetDemand
            unmetDemand1 = retailer1.GetUnmetDemand_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])
            unmetDemand2 = retailer2.GetUnmetDemand_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])                                                            
            
            
            # get Amount Trans To
            amountTransTo1 = 0
            amountTransTo2 = 0
            
            
            # save results for printing out
            retailer1.inventoryLevelBeginningList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelBeginningList.append(retailer2.inventoryLevel)
            
            
            # inventory update
            retailer1.InventoryUpdate_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])           
            retailer2.InventoryUpdate_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])
            
            
            # save results for printing out
            retailer1.inventoryLevelRemainingList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelRemainingList.append(retailer2.inventoryLevel)
            
            retailer1.realizedRewardList.append(currentReward1)
            retailer2.realizedRewardList.append(currentReward2)
            
            retailer1.orderAmountList.append(retailer1.orderAmount)
            retailer2.orderAmountList.append(retailer2.orderAmount)
            
            retailer1.unmetDemandList.append(unmetDemand1)
            retailer2.unmetDemandList.append(unmetDemand2)
            
            
            retailer1.amountTransToList.append(amountTransTo1)
            retailer2.amountTransToList.append(amountTransTo2)
            
            
            # t <- t+1
            t += 1
    
    
    
    def Run_centralized(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, a1, a2, x1, x2, s,
            seedRndEGreedy, seedRndAction1, seedRndAction2, seedRndX1, seedRndX2, seedRndPi, initialInventoryLevel1, initialInventoryLevel2):
        # for centralized Q-learning
        
        rndEGreedyGenerator = random.Random(seedRndEGreedy)
        rndAction1Generator = random.Random(seedRndAction1)
        rndAction2Generator = random.Random(seedRndAction2)
        rndX1Generator = random.Random(seedRndX1)
        rndX2Generator = random.Random(seedRndX2)
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel1
        retailer2.inventoryLevel = initialInventoryLevel2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndEGreedy = rndEGreedyGenerator.random()
            rndAction1 = rndAction1Generator.random()
            rndAction2 = rndAction2Generator.random()
            rndX1 = rndX1Generator.random()
            rndX2 = rndX2Generator.random()
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            if rndEGreedy <= self.exploreFactor:
                # a1
                for i in range(len(a1.action)):
                    if (((1.0 / len(a1.action)) * i) <= rndAction1) and (rndAction1 < ((1.0 / len(a1.action)) * (i + 1))):
                        self.a1_Index = i
                        break
                
                # a2
                for i in range(len(a2.action)):
                    if (((1.0 / len(a2.action)) * i) <= rndAction2) and (rndAction2 < ((1.0 / len(a2.action)) * (i + 1))):
                        self.a2_Index = i
                        break
                    
                # x1
                for i in range(len(x1.action)):
                    if (((1.0 / len(x1.action)) * i) <= rndX1) and (rndX1 < ((1.0 / len(x1.action)) * (i + 1))):
                        self.x1_Index = i
                        break
                
                # x2
                for i in range(len(x2.action)):
                    if (((1.0 / len(x2.action)) * i) <= rndX2) and (rndX2 < ((1.0 / len(x2.action)) * (i + 1))):
                        self.x2_Index = i
                        break
                
                
            else:
                (self.a1_Index, self.a2_Index, self.x1_Index, self.x2_Index) = self.GetIndexFromInverseCumulativePi_centralized(currentStateIndex1, currentStateIndex2, rndPi, a1, a2, x1, x2)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            
            # save transshipment amount (not actual transshipment amount!)
            retailer1.xAmount = x1.action[self.x1_Index]
            retailer2.xAmount = x2.action[self.x2_Index]
            
            # save effective transshipment amount
            retailer1.yAmount = min(max(retailer1.xAmount - retailer2.xAmount, 0), retailer1.inventoryLevel)    # y_ij
            retailer2.yAmount = min(max(retailer2.xAmount - retailer1.xAmount, 0), retailer2.inventoryLevel)    # y_ji
            
            
            # get R(s1, s2, a1, a2, x1, x2)
            currentReward = self.GetRealizedReward_centralized(demand1.realizedArray[t], demand2.realizedArray[t], supplyLoss1.realizedArray[t], supplyLoss2.realizedArray[t], 
                                                               retailer1, retailer2)
            
            
            # get next inventory level
            nextInventoryLevel1 = retailer1.GetInventoryUpdate_centralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t], retailer2.yAmount)
            nextInventoryLevel2 = retailer2.GetInventoryUpdate_centralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t], retailer1.yAmount)
            
            # find next state indices
            (nextStateIndex1, nextStateIndex2) = self.GetStateIndex(nextInventoryLevel1, nextInventoryLevel2, s)
            
            
            
            # update Q(s1, s2, a1, a2, x1, x2)
            self.q_centralized[currentStateIndex1, currentStateIndex2, self.a1_Index, self.a2_Index, self.x1_Index, self.x2_Index] = (((1.0 - self.learningRate) * self.q_centralized[currentStateIndex1, currentStateIndex2, self.a1_Index, self.a2_Index, self.x1_Index, self.x2_Index]) 
            + (self.learningRate * (currentReward + (self.discountFactor * self.GetMaxDiscountedFutureQValue_centralized(nextStateIndex1, nextStateIndex2)))))
            
            
            
            
            # update pi(s1, s2, a1, a2, x1, x2) by using Softmax rule
            self.UpdatePi_centralized(currentStateIndex1, currentStateIndex2, len(a1.action), len(a2.action), len(x1.action), len(x2.action))


            # learning rate update
            self.learningRate *= self.decayFactor
            
            
            # inventory update
            retailer1.InventoryUpdate_centralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t], retailer2.yAmount)           
            retailer2.InventoryUpdate_centralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t], retailer1.yAmount)
            
            
            # t <- t+1
            t += 1
    
    
    
    def Run_centralized_New(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s,
            seedRndEGreedy, seedRndAction1, seedRndAction2, seedRndO, seedRndPi, initialInventoryLevel1, initialInventoryLevel2, riskSensitivity1, riskSensitivity2):
        # for centralized Q-learning
        
        rndEGreedyGenerator = random.Random(seedRndEGreedy)
        rndAction1Generator = random.Random(seedRndAction1)
        rndAction2Generator = random.Random(seedRndAction2)
        rndOGenerator = random.Random(seedRndO)
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel1
        retailer2.inventoryLevel = initialInventoryLevel2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndEGreedy = rndEGreedyGenerator.random()
            rndAction1 = rndAction1Generator.random()
            rndAction2 = rndAction2Generator.random()
            rndO = rndOGenerator.random()
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel * riskSensitivity1, retailer2.inventoryLevel * riskSensitivity2, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            if rndEGreedy <= self.exploreFactor:
                # a1
                for i in range(len(a1.action)):
                    if (((1.0 / len(a1.action)) * i) <= rndAction1) and (rndAction1 < ((1.0 / len(a1.action)) * (i + 1))):
                        self.a1_Index = i
                        break
                
                # a2
                for i in range(len(a2.action)):
                    if (((1.0 / len(a2.action)) * i) <= rndAction2) and (rndAction2 < ((1.0 / len(a2.action)) * (i + 1))):
                        self.a2_Index = i
                        break
                    
                # o
                for i in range(len(o.action)):
                    if (((1.0 / len(o.action)) * i) <= rndO) and (rndO < ((1.0 / len(o.action)) * (i + 1))):
                        self.o_Index = i
                        break
                
                
                
            else:
                (self.a1_Index, self.a2_Index, self.o_Index) = self.GetIndexFromInverseCumulativePi_centralized_New(currentStateIndex1, currentStateIndex2, rndPi, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R(s1, s2, a1, a2, o)
            currentReward1 = retailer1.GetRealizedReward_centralized_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            currentReward2 = retailer2.GetRealizedReward_centralized_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)
            
            currentReward = currentReward1 + currentReward2
            
            # get next inventory level
            nextInventoryLevel1 = retailer1.GetInventoryUpdate_centralized_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)
            nextInventoryLevel2 = retailer2.GetInventoryUpdate_centralized_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            # find next state indices
            (nextStateIndex1, nextStateIndex2) = self.GetStateIndex(nextInventoryLevel1 * riskSensitivity1, nextInventoryLevel2 * riskSensitivity2, s)
            
            
            
            # update Q(s1, s2, a1, a2, o)
            self.q_centralized_new[currentStateIndex1, currentStateIndex2, self.a1_Index, self.a2_Index, self.o_Index] = (((1.0 - self.learningRate) * self.q_centralized_new[currentStateIndex1, currentStateIndex2, self.a1_Index, self.a2_Index, self.o_Index]) 
            + (self.learningRate * (currentReward + (self.discountFactor * self.GetMaxDiscountedFutureQValue_centralized_new(nextStateIndex1, nextStateIndex2)))))
            
            
            # update Q count (state-action pair count)
            self.q_cnt[currentStateIndex1, currentStateIndex2, self.a1_Index, self.a2_Index, self.o_Index] = self.q_cnt[currentStateIndex1, currentStateIndex2, self.a1_Index, self.a2_Index, self.o_Index]  + 1
            
            
            # update pi(s1, s2, a1, a2, x1, x2) by using Softmax rule
            self.UpdatePi_centralized_new(currentStateIndex1, currentStateIndex2, len(a1.action), len(a2.action), len(o.action))


            # learning rate update
            self.learningRate *= self.decayFactor
            
            
            # inventory update
            retailer1.InventoryUpdate_centralized_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_centralized_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            
            # t <- t+1
            t += 1
    
    
            
            
            
    def Test_centralized(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, a1, a2, x1, x2, s, seedRndPi, initialInventoryLevel_1, initialInventoryLevel_2):
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel_1
        retailer2.inventoryLevel = initialInventoryLevel_2
        
        t = 0
        while(t < maxT):
            # generate random number
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            
            (self.a1_Index, self.a2_Index, self.x1_Index, self.x2_Index) = self.GetIndexFromInverseCumulativePi_centralized(currentStateIndex1, currentStateIndex2, rndPi, a1, a2, x1, x2)
            
            
            
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            # save transshipment amount (not actual transshipment amount!)
            retailer1.xAmount = x1.action[self.x1_Index]
            retailer2.xAmount = x2.action[self.x2_Index]
            
            
            # save effective transshipment amount
            retailer1.yAmount = min(max(retailer1.xAmount - retailer2.xAmount, 0), retailer1.inventoryLevel)    # y_ij
            retailer2.yAmount = min(max(retailer2.xAmount - retailer1.xAmount, 0), retailer2.inventoryLevel)    # y_ji
            
            
            # get R(s1, s2, a1, a2, x1, x2)
            currentReward1 = retailer1.GetRealizedReward_centralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t], retailer2)
            currentReward2 = retailer2.GetRealizedReward_centralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t], retailer1)
            
            # get UnmetDemand
            unmetDemand1 = retailer1.GetUnmetDemand_centralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t], retailer2)
            unmetDemand2 = retailer2.GetUnmetDemand_centralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t], retailer1)
            
            
            
            
            # save results for printing out
            retailer1.inventoryLevelBeginningList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelBeginningList.append(retailer2.inventoryLevel)
            
            
            
            # inventory update
            retailer1.InventoryUpdate_centralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t], retailer2.yAmount)           
            retailer2.InventoryUpdate_centralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t], retailer1.yAmount)
            
            
            
            # save results for printing out
            retailer1.inventoryLevelRemainingList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelRemainingList.append(retailer2.inventoryLevel)
            
            
            retailer1.realizedRewardList.append(currentReward1)
            retailer2.realizedRewardList.append(currentReward2)
            
            
            retailer1.orderAmountList.append(retailer1.orderAmount)
            retailer2.orderAmountList.append(retailer2.orderAmount)
            
            retailer1.yAmountList.append(retailer1.yAmount)
            retailer2.yAmountList.append(retailer2.yAmount)
            
            retailer1.unmetDemandList.append(unmetDemand1)
            retailer2.unmetDemandList.append(unmetDemand2)
            
            retailer1.amountTransToList.append(retailer1.yAmount)
            retailer2.amountTransToList.append(retailer2.yAmount)
            
            # t <- t+1
            t += 1   
            
            
    
    
            
            
    def Test_centralized_New(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, parent, a1, a2, o, s,
            seedRndPi, initialInventoryLevel1, initialInventoryLevel2):
        
        
        
        # for centralized Q-learning
        
        rndPiGenerator = random.Random(seedRndPi)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel1
        retailer2.inventoryLevel = initialInventoryLevel2
        
        t = 0
        while(t < maxT):
            # generate random numbers
            rndPi = rndPiGenerator.random()
            
            # find the current stateIndices
            currentStateIndices = self.GetStateIndex(retailer1.inventoryLevel, retailer2.inventoryLevel, s)
            currentStateIndex1 = currentStateIndices[0]
            currentStateIndex2 = currentStateIndices[1]
            
            
            (self.a1_Index, self.a2_Index, self.o_Index) = self.GetIndexFromInverseCumulativePi_centralized_New(currentStateIndex1, currentStateIndex2, rndPi, a1, a2, o)
                
                
            # save order amount
            retailer1.orderAmount = a1.action[self.a1_Index]
            retailer2.orderAmount = a2.action[self.a2_Index]
            
            
            # save transshipment amount (not actual transshipment amount!)
            parent.transshipmentAmount = o.action[self.o_Index]
            
            
            # get R(s1, s2, a1, a2, o)
            currentReward1 = retailer1.GetRealizedReward_centralized_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            currentReward2 = retailer2.GetRealizedReward_centralized_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)
            
            # get UnmetDemand
            unmetDemand1 = retailer1.GetUnmetDemand_centralized_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], 
                                                         o.action[self.o_Index][0], retailer2.unitTransshipmentRevenue, retailer2.inventoryLevel)
            
            unmetDemand2 = retailer2.GetUnmetDemand_centralized_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], 
                                                         o.action[self.o_Index][1], retailer1.unitTransshipmentRevenue, retailer1.inventoryLevel)
            
            
            # get Amount Trans To
            amountTransTo1 = min(retailer1.inventoryLevel, o.action[self.o_Index][0])
            amountTransTo2 = min(retailer2.inventoryLevel, o.action[self.o_Index][1])
            
            
            
            # save results for printing out
            retailer1.inventoryLevelBeginningList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelBeginningList.append(retailer2.inventoryLevel)
            
            
            # inventory update
            retailer1.InventoryUpdate_centralized_new(demand1.realizedArray[t], supplyLoss1.realizedArray[t], o.action[self.o_Index][1], o.action[self.o_Index][0], retailer2.inventoryLevel)           
            retailer2.InventoryUpdate_centralized_new(demand2.realizedArray[t], supplyLoss2.realizedArray[t], o.action[self.o_Index][0], o.action[self.o_Index][1], retailer1.inventoryLevel)
            
            # save results for printing out
            retailer1.inventoryLevelRemainingList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelRemainingList.append(retailer2.inventoryLevel)
            
            
            retailer1.realizedRewardList.append(currentReward1)
            retailer2.realizedRewardList.append(currentReward2)
            
            
            retailer1.orderAmountList.append(retailer1.orderAmount)
            retailer2.orderAmountList.append(retailer2.orderAmount)
            
            parent.transshipmentAmountList.append(parent.transshipmentAmount)
            
            retailer1.unmetDemandList.append(unmetDemand1)
            retailer2.unmetDemandList.append(unmetDemand2)
            
            retailer1.amountTransToList.append(amountTransTo1)
            retailer2.amountTransToList.append(amountTransTo2)
            
            # t <- t+1
            t += 1
     
     
    def Run_baseStockGA(self, maxT, demand1, demand2, supplyLoss1, supplyLoss2, retailer1, retailer2, initialInventoryLevel_1, initialInventoryLevel_2, 
                                    baseStockUpperBound, baseStockLowerBound1, baseStockLowerBound2, populationSize, generationSize, supplierCapacity,
                                    seedCR1, seedCR2, seedMR1, seedMR2, seedOne1, seedOne2, seedMut1, seedMut2, seedInitialPop1, seedInitialPop2,
                                    seedMating1, seedMating2, CR, MR):
    

        rndCR1Generator = random.Random(seedCR1)
        rndCR2Generator = random.Random(seedCR2)
        rndMR1Generator = random.Random(seedMR1)
        rndMR2Generator = random.Random(seedMR2)
        rndOne1Generator = random.Random(seedOne1)
        rndOne2Generator = random.Random(seedOne2)
        rndMut1Generator = random.Random(seedMut1)
        rndMut2Generator = random.Random(seedMut2)
        rndMating1Generator = random.Random(seedMating1)
        rndMating2Generator = random.Random(seedMating2)
        rndInitialPop1Generator = random.Random(seedInitialPop1)
        rndInitialPop2Generator = random.Random(seedInitialPop2)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel_1
        retailer2.inventoryLevel = initialInventoryLevel_2
        
        
        # initialize population
        initialPopulation1 = []
        initialPopulation2 = []
        for i in range(0, populationSize):
            initialPopulation1.append(Population(baseStockUpperBound, baseStockLowerBound1, rndInitialPop1Generator, maxT))
            initialPopulation2.append(Population(baseStockUpperBound, baseStockLowerBound2, rndInitialPop2Generator, maxT))
        
        
        for no_gen in tqdm(range(generationSize)):
             
            
            # calculate the fitness value for retailer1
            for i in range(len(initialPopulation1)):
                retailer1.inventoryLevel = initialInventoryLevel_1
                retailer2.inventoryLevel = initialInventoryLevel_2

                for t in range(len(initialPopulation1[i].chromosome)):
                    # save order amount
                    retailer1.orderAmount =  min(max(initialPopulation1[i].chromosome[t] - retailer1.inventoryLevel, 0), supplierCapacity)
                    retailer2.orderAmount =  min(max(initialPopulation2[i].chromosome[t] - retailer2.inventoryLevel, 0), supplierCapacity)
                    
                    # set fitness value (+= reward)
                    initialPopulation1[i].SetFitnessValue(initialPopulation1[i].fitnessValue + retailer1.GetRealizedReward_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t]))
                    initialPopulation2[i].SetFitnessValue(initialPopulation2[i].fitnessValue + retailer2.GetRealizedReward_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t]))
                    
                    
                    # inventory update
                    retailer1.InventoryUpdate_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])           
                    retailer2.InventoryUpdate_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])
            
            
                
            # set the cumulative probability based on the fitness value
            sumFitnessValue1 = 0.0
            sumFitnessValue2 = 0.0
            for i in range(len(initialPopulation1)):
                sumFitnessValue1 = sumFitnessValue1 + initialPopulation1[i].fitnessValue
                sumFitnessValue2 = sumFitnessValue2 + initialPopulation2[i].fitnessValue
                
            for i in range(len(initialPopulation1)):
                if i == 0:
                    initialPopulation1[i].cumProbability = initialPopulation1[i].fitnessValue / sumFitnessValue1
                    initialPopulation2[i].cumProbability = initialPopulation2[i].fitnessValue / sumFitnessValue2
                else:
                    initialPopulation1[i].cumProbability = initialPopulation1[i-1].cumProbability + (initialPopulation1[i].fitnessValue / sumFitnessValue1)
                    initialPopulation2[i].cumProbability = initialPopulation2[i-1].cumProbability + (initialPopulation2[i].fitnessValue / sumFitnessValue2)
    
    
    
            # create Mating pool
            matingPopulation1 = []
            matingPopulation2 = []
            
            cnt1 = 0
            while(cnt1 < len(initialPopulation1)):
                rndMating1 = rndMating1Generator.random()
                for i in range(0, len(initialPopulation1)):
                    if i == 0:
                        if ((rndMating1 <= initialPopulation1[i].cumProbability) and (initialPopulation1[i].mating == False)):
                            matingPop = copy.deepcopy(initialPopulation1[i])
                            matingPopulation1.append(matingPop)
                            initialPopulation1[i].mating = True
                            cnt1 = cnt1 + 1  
                    else:
                        if ((rndMating1 > initialPopulation1[i - 1].cumProbability) and (rndMating1 <= initialPopulation1[i].cumProbability) and (initialPopulation1[i].mating == False)):
                                matingPop = copy.deepcopy(initialPopulation1[i])
                                matingPopulation1.append(matingPop)
                                initialPopulation1[i].mating = True
                                cnt1 = cnt1 + 1  
                            
                    
            cnt2 = 0
            while(cnt2 < len(initialPopulation2)):
                rndMating2 = rndMating2Generator.random()
                for i in range(0, len(initialPopulation2)):
                    if i == 0:
                        if ((rndMating2 <= initialPopulation2[i].cumProbability) and (initialPopulation2[i].mating == False)):
                            matingPop = copy.deepcopy(initialPopulation2[i])
                            matingPopulation2.append(matingPop)
                            initialPopulation2[i].mating = True
                            cnt2 = cnt2 + 1  
                    else:
                        if ((rndMating2 > initialPopulation2[i - 1].cumProbability) and (rndMating2 <= initialPopulation2[i].cumProbability) and (initialPopulation2[i].mating == False)):
                                matingPop = copy.deepcopy(initialPopulation2[i])
                                matingPopulation2.append(matingPop)
                                initialPopulation2[i].mating = True
                                cnt2 = cnt2 + 1                      
                    
                        
            # Crossover
            intermediatePopulation1 = []
            intermediatePopulation2 = []
            for i in range(len(matingPopulation1)):
                if i % 2 == 0:
                    rndC1 = rndCR1Generator.random()
                    if rndC1 <= CR:
                        rndOne1 = rndOne1Generator.randint(0, len(matingPopulation1[i].chromosome) - 2)
                        tempChromosomeI = matingPopulation1[i].chromosome[rndOne1:]
                        tempChromosomeI_1 = matingPopulation1[i + 1].chromosome[rndOne1:]
                        
                        del matingPopulation1[i].chromosome[rndOne1:]
                        del matingPopulation1[i + 1].chromosome[rndOne1:]
                        
                        matingPopulation1[i].chromosome.extend(tempChromosomeI_1)
                        matingPopulation1[i + 1].chromosome.extend(tempChromosomeI)
                        
                    intermediatePopI = copy.deepcopy(matingPopulation1[i])
                    intermediatePopulation1.append(intermediatePopI)
                    
                    intermediatePopI_1 = copy.deepcopy(matingPopulation1[i + 1])
                    intermediatePopulation1.append(intermediatePopI_1)
                    
                        
                        
            for i in range(len(matingPopulation2)):
                if i % 2 == 0:
                    rndC2 = rndCR2Generator.random()
                    if rndC2 <= CR:
                        rndOne2 = rndOne2Generator.randint(0, len(matingPopulation2[i].chromosome) - 2)
                        tempChromosomeI = matingPopulation2[i].chromosome[rndOne2:]
                        tempChromosomeI_1 = matingPopulation2[i + 1].chromosome[rndOne2:]
                        
                        del matingPopulation2[i].chromosome[rndOne2:]
                        del matingPopulation2[i + 1].chromosome[rndOne2:]
                        
                        matingPopulation2[i].chromosome.extend(tempChromosomeI_1)
                        matingPopulation2[i + 1].chromosome.extend(tempChromosomeI)
                        
                    intermediatePopI = copy.deepcopy(matingPopulation2[i])
                    intermediatePopulation2.append(intermediatePopI)
                    
                    intermediatePopI_1 = copy.deepcopy(matingPopulation2[i + 1])
                    intermediatePopulation2.append(intermediatePopI_1)
                
                
                        
            # mutation
            for i in range(len(intermediatePopulation1)):
                for j in range(len(intermediatePopulation1[i].chromosome)):
                    rndMR1 = rndMR1Generator.random()
                    if rndMR1 <= MR:
                        s_new = round((intermediatePopulation1[i].chromosome[j] * (1 - 0.2)) + (intermediatePopulation1[i].chromosome[j] * 2 * 0.2 * rndMut1Generator.random()))
                        intermediatePopulation1[i].chromosome[j] = s_new
                
            for i in range(len(intermediatePopulation2)):
                for j in range(len(intermediatePopulation2[i].chromosome)):
                    rndMR2 = rndMR2Generator.random()
                    if rndMR2 <= MR:
                        s_new = round((intermediatePopulation2[i].chromosome[j] * (1 - 0.2)) + (intermediatePopulation2[i].chromosome[j] * 2 * 0.2 * rndMut2Generator.random()))
                        intermediatePopulation2[i].chromosome[j] = s_new
            
            # initialize fitnessValue for intermediatePopulation
            for i in range(len(intermediatePopulation1)):
                intermediatePopulation1[i].fitnessValue = 0.0   
                intermediatePopulation2[i].fitnessValue = 0.0
                
                        
            # copy retailer
            copyRetailer1 = copy.deepcopy(retailer1)
            copyRetailer2 = copy.deepcopy(retailer2)
            copyRetailer1.inventoryLevel = initialInventoryLevel_1
            copyRetailer2.inventoryLevel = initialInventoryLevel_2
            
            # calculate the fitness value for copyRetailer
            for i in range(len(intermediatePopulation1)):
                copyRetailer1.inventoryLevel = initialInventoryLevel_1
                copyRetailer2.inventoryLevel = initialInventoryLevel_2
            
                for t in range(len(intermediatePopulation1[i].chromosome)):
                    # save order amount
                    copyRetailer1.orderAmount =  min(max(intermediatePopulation1[i].chromosome[t] - copyRetailer1.inventoryLevel, 0), supplierCapacity)
                    copyRetailer2.orderAmount =  min(max(intermediatePopulation2[i].chromosome[t] - copyRetailer2.inventoryLevel, 0), supplierCapacity)
                    
                    # set fitness value (+= reward)
                    intermediatePopulation1[i].SetFitnessValue(intermediatePopulation1[i].fitnessValue + copyRetailer1.GetRealizedReward_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t]))
                    intermediatePopulation2[i].SetFitnessValue(intermediatePopulation2[i].fitnessValue + copyRetailer2.GetRealizedReward_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t]))
                    
                    
                    # inventory update
                    copyRetailer1.InventoryUpdate_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])           
                    copyRetailer2.InventoryUpdate_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])
                    
                    
            # elite selection
            initialPopulation1.extend(intermediatePopulation1)
            initialPopulation2.extend(intermediatePopulation2)
            
            initialPopulation1 = sorted(initialPopulation1)
            initialPopulation2 = sorted(initialPopulation2)
            
            del initialPopulation1[len(intermediatePopulation1):]
            del initialPopulation2[len(intermediatePopulation2):]
            
            # initialize fitness value
            for i in range(len(initialPopulation1)):
                initialPopulation1[i].fitnessValue = 0.0
                initialPopulation2[i].fitnessValue = 0.0
                initialPopulation1[i].cumProbability = 0.0
                initialPopulation2[i].cumProbability = 0.0
                initialPopulation1[i].mating = False
                initialPopulation2[i].mating = False
                
            
            
            
        # Get Best Solution
        retailer1.chromosome = []
        retailer2.chromosome = []
        retailer1.chromosome.extend(initialPopulation1[0].chromosome)
        retailer2.chromosome.extend(initialPopulation2[0].chromosome)
        
        # initialize inventory
        retailer1.inventoryLevel = initialInventoryLevel_1
        retailer2.inventoryLevel = initialInventoryLevel_2
        
        t = 0
        while(t < maxT):
                
            # save order amount
            retailer1.orderAmount =  min(max(retailer1.chromosome[t] - retailer1.inventoryLevel, 0), supplierCapacity)
            retailer2.orderAmount =  min(max(retailer2.chromosome[t] - retailer2.inventoryLevel, 0), supplierCapacity)
            
            # get R1
            currentReward1 = retailer1.GetRealizedReward_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])
            # get R2
            currentReward2 = retailer2.GetRealizedReward_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t]) 
            
            
            # get UnmetDemand
            unmetDemand1 = retailer1.GetUnmetDemand_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])
            unmetDemand2 = retailer2.GetUnmetDemand_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])                                                            
            
            
            # get Amount Trans To
            amountTransTo1 = 0
            amountTransTo2 = 0
            
            
            # save results for printing out
            retailer1.inventoryLevelBeginningList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelBeginningList.append(retailer2.inventoryLevel)
            
            
            # inventory update
            retailer1.InventoryUpdate_decentralized(demand1.realizedArray[t], supplyLoss1.realizedArray[t])           
            retailer2.InventoryUpdate_decentralized(demand2.realizedArray[t], supplyLoss2.realizedArray[t])
                    
            
            
            # save results for printing out
            retailer1.inventoryLevelRemainingList.append(retailer1.inventoryLevel)
            retailer2.inventoryLevelRemainingList.append(retailer2.inventoryLevel)
            
            retailer1.realizedRewardList.append(currentReward1)
            retailer2.realizedRewardList.append(currentReward2)
            
            retailer1.orderAmountList.append(retailer1.orderAmount)
            retailer2.orderAmountList.append(retailer2.orderAmount)
            
            retailer1.unmetDemandList.append(unmetDemand1)
            retailer2.unmetDemandList.append(unmetDemand2)
            
            
            retailer1.amountTransToList.append(amountTransTo1)
            retailer2.amountTransToList.append(amountTransTo2)
            
            
            # t <- t+1
            t += 1
        
        
        
        
   
            