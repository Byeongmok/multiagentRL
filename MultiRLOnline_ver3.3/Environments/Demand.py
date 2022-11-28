import numpy as np
import random
from scipy.stats import norm


class Demand():
    def __init__(self, seedDemandMean, seedDemandStdDev, seedDemandRealization, maxT, meanMin, meanMax, stdMin, stdMax):
        self.meanList = []
        self.realizedArray = np.zeros(shape=[maxT])
        self.sigmaList = []
        self.seedDemandRealization = seedDemandRealization
        
        random.seed(seedDemandMean)
        for i in range(maxT):
            self.meanList.append(random.randint(meanMin, meanMax))
            
        random.seed(seedDemandStdDev)
        for i in range(maxT):
            self.sigmaList.append(random.randint(stdMin, stdMax))
            
        random.seed(seedDemandRealization)
        for i in range(maxT):
            realizedDemandValue = self.getRealizedDemand(i, random.random())
            self.realizedArray[i] = realizedDemandValue if realizedDemandValue >= 0 else 0
        
            
    def getRealizedDemand(self, t, randomNumber):
        value = norm.ppf(randomNumber, loc=self.meanList[t], scale = self.sigmaList[t])
        
        return round(value)
        