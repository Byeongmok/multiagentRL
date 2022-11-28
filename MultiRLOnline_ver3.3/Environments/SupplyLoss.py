import numpy as np
import random
from scipy.stats import norm


class SupplyLoss():
    def __init__(self, seedSupplyLossMean, seedSupplyLossStdDev, seedSupplyLossRealization, maxT, meanMin, meanMax, stdMin, stdMax):
        self.meanList = []
        self.realizedArray = np.zeros(shape=[maxT])
        self.sigmaList = []
        self.seedSupplyLossRealization = seedSupplyLossRealization
        
        random.seed(seedSupplyLossMean)
        for i in range(maxT):
            self.meanList.append(random.randint(meanMin, meanMax))
            
        random.seed(seedSupplyLossStdDev)
        for i in range(maxT):
            self.sigmaList.append(random.randint(stdMin, stdMax))
            
        random.seed(seedSupplyLossRealization)
        for i in range(maxT):
            realizedSupplyLossValue = self.getRealizedSupplyLoss(i, random.random())
            self.realizedArray[i] = realizedSupplyLossValue if realizedSupplyLossValue >= 0 else 0
        
            
    def getRealizedSupplyLoss(self, t, randomNumber):
        value = norm.ppf(randomNumber, loc=self.meanList[t], scale = self.sigmaList[t])
        
        return round(value)
        