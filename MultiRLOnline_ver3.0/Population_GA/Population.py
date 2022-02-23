class Population():
    def __init__(self, baseStockUpperBound, baseStockLowerBound, rndInitialPop, maxT):
        self.baseStockUpperBound = baseStockUpperBound
        self.baseStockLowerBound = baseStockLowerBound
        self.rndInitialPop = rndInitialPop
        self.maxT = maxT
        self.fitnessValue = 0.0
        self.cumProbability = 0.0
        self.mating = False
        
        self.chromosome = []
        
        if rndInitialPop != 9999:
            for i in range(0, maxT):
                self.chromosome.append(rndInitialPop.randint(baseStockLowerBound, baseStockUpperBound))
            
    def SetFitnessValue(self, fitnessValue):
        self.fitnessValue = fitnessValue
        
    def __lt__(self, other):
        return self.fitnessValue > other.fitnessValue