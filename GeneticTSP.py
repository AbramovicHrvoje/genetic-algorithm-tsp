import numpy as np
from numpy import inf
from random import random
from random import randrange
from random import choice
from random import uniform
from random import choices
import time
import statistics as stats

## HRVOJE ABRAMOVIC r0829189

# THE IDEA BEHIND THIS APPROACH
## By creating a lot of static methods grouped by classes, certain functionalities are being modularized,
## so that changing certain parts of the algorithm is done in Strategy design pattern style (sort of)



## CHROMOSOME: permuation of 0-indexed cities - but - city 0 is not in permutation
## it always starts and ends with 0, so it brings no info, but might complicate mutation and crossover

##DISTANCES
# static class that holds distanceMatrix (distances between cities) -- so it can be globally accesbile without global object
# and all the methods regarding the distance matrix
class Distances:
    distanceMatrix = None
    numCities = None
    numberOfInfs = None
    ## method to set the distance matrix
    @staticmethod
    def setMatrix(distanceMatrix):
        Distances.distanceMatrix = distanceMatrix
        Distances.numCities = Distances.distanceMatrix.shape[0]
        Distances.countInfs()

    ## use this method to get distances between 2 cities
    @staticmethod
    def getDistance(cityA, cityB):
        return Distances.distanceMatrix[cityA][cityB]

    @staticmethod
    def numberOfCities():
        return Distances.numCities
    
    ## get a set of cities connected to city city
    @staticmethod
    def getAdjCities(city):
        adj = set()
        for i in range(Distances.numberOfCities()):
            if Distances.getDistance(city, i) < inf and city != i:
                adj.add(i)
                
        return adj
    
    ## get a list of cities connected to city city but sorted
    @staticmethod
    def getSortedAdjList(city):
        adj = []
        for i in range(Distances.numberOfCities()):
            if Distances.getDistance(city, i)  < inf and city != i:
                adj.append((i, Distances.getDistance(city, i)))
        
        adj = sorted(adj, key = lambda x: x[1])
        adj = [x[0] for x in adj]
        return adj
    
    ## get numpy array path and evaluate it's cost
    @staticmethod
    def pathCost(path):
        cost = Distances.getDistance(0, path[0])

        for i in range(1, Distances.numberOfCities()-1):
            cost += Distances.getDistance(path[i - 1], path[i])

        cost += Distances.getDistance(path[-1], 0)
        
        return cost
    
    @staticmethod
    def randomPermutation():
        return np.random.permutation(range(1, Distances.numberOfCities()))
    
    ## compute Hamming distance between two paths
    @staticmethod
    def hammingDistance(path1, path2):
        return np.count_nonzero(path1-path2)
    
    @staticmethod
    def countInfs():
        Distances.numberOfInfs = 0
        for i in range(Distances.numCities):
            for j in range(Distances.numCities):
                if Distances.distanceMatrix[i][j] == inf:
                    Distances.numberOfInfs+=1
                    
        return

## ClASS OF INITIALIZATION HEURISTICS
## EVERY HEURISTIC IS A STATIC METHOD AND RETURNS A np.array  
## if any of the heuristics fail (stuck in impossible position, only possible in examples that have inf)  -- returns a random permutation, backtracking would be slow  
class InitHeuristics:

    ## Nearest Neighbour with a probabilistic twist
    ## In every step there is a unfair coin toss, and depending on the result the next city is being selected randomly (uniform)
    ## or with a probability that is proportinal to how close the city is
    @staticmethod
    def stohasticNN(prob):
        visited = set()

        path = np.zeros((Distances.numberOfCities()-1, ), dtype = int) # exclude 0 city
        
        currentCity = 0
        visited.add(0)
        index = 0
        
        while index < Distances.numberOfCities()-1:
            if random() < prob:
                nextCity = -1
                adjCities = Distances.getAdjCities(currentCity)
                possibleCities = []
                citiesWeights = []
                for city in adjCities:
                    if city not in visited:
                        possibleCities.append(city)
                        citiesWeights.append(1/Distances.getDistance(currentCity, city))
                if len(possibleCities) == 0:
                    path = Distances.randomPermutation()
                    return path
                nextCity = choices(possibleCities, weights = citiesWeights, k = 1)[0]
                currentCity = nextCity
                
            else:
                possibleCitiesSet = Distances.getAdjCities(currentCity) - visited
                if not bool(possibleCitiesSet):
                    path = Distances.randomPermutation()
                    return path
                currentCity = choice(tuple(possibleCitiesSet))
            
            
            path[index] = currentCity
            visited.add(currentCity)
            index+=1
    
        return path
    
    ## Nearest Neighbour with a probabilistic twist v.2
    ## In every step there is a unfair coin toss, and depending on the result the next city is being selected randomly (uniform)
    ## or picks the next best neighbour -- much more robust, but shows better performance
    @staticmethod
    def coinFlipNN(prob):
        visited = set()

        path = np.zeros((Distances.numberOfCities()-1, ), dtype = int) # exclude 0 city
        
        currentCity = 0
        visited.add(0)
        index = 0
        
        while index < Distances.numberOfCities()-1:
            if random() < prob:
                nextCity = -1
                possibleCities = Distances.getSortedAdjList(currentCity)
                for city in possibleCities:
                    if city not in visited:
                        nextCity = city
                        break
                if nextCity == -1:
                    path = Distances.randomPermutation()
                    return path
                currentCity = nextCity
                
            else:
                possibleCitiesSet = Distances.getAdjCities(currentCity) - visited
                if not bool(possibleCitiesSet):
                    path = Distances.randomPermutation()
                    return path
                currentCity = choice(tuple(possibleCitiesSet))
            
            
            path[index] = currentCity
            visited.add(currentCity)
            index+=1
    
        return path
    
    ## Random walk that doesn't include inf values
    @staticmethod
    def randomWalk():
        visited = set()

        path = np.zeros((Distances.numberOfCities()-1, ), dtype = int) # exclude 0 city
        
        currentCity = 0
        visited.add(0)
        index = 0
        
        while index < Distances.numberOfCities()-1:
            possibleCitiesSet = Distances.getAdjCities(currentCity) - visited
            if not bool(possibleCitiesSet):
                path = Distances.randomPermutation()
                break
            currentCity = choice(tuple(possibleCitiesSet)) 
            path[index] = currentCity
            visited.add(currentCity)
            index+=1
    
        return path
    

    ## Sample random numOfSamples times and take the best one
    @staticmethod
    def monteCarloInit(numOfSamples):
        
        bestPath = None
        bestCost = inf
        
        for i in range(numOfSamples):
            rndPath = Distances.randomPermutation()
            currentCost = Distances.pathCost(rndPath)
            if bestPath is None or bestCost > currentCost:
                bestCost = currentCost
                bestPath = rndPath
        
        return bestPath


    ## Initialization that picks a random city and then lists all cities regarding their distance to that city
    @staticmethod
    def mixtureInit():
        
        path = np.zeros((Distances.numberOfCities()-1, ), dtype = int)
        
        
        firstCity = randrange(1, Distances.numberOfCities())
        path[0] = firstCity
        adjCities = Distances.getSortedAdjList(firstCity)
        index = 1
        
        if len(adjCities) != Distances.numberOfCities()-1:
            #print("NOT GOOD")
            return InitHeuristics.randomWalk()
        
        for city in adjCities:
            if city == 0:
                continue
            path[index] = city
            index+=1
        
        return path
        
## POPULATION INITALIZATIONS
## serious of functions that 
class PopulationInitializations:
    
    ## Linear Monte Carlo - 
    @staticmethod
    def LinearMonteCarlo(size):
        initPop = []
        for i in range(size):
            initPop.append(Solution(permutation = InitHeuristics.monteCarloInit((i+5))))
            
        return initPop
    
    ## Serious of variations of NN with different coint toss probabilites + monte carlo at the end
    @staticmethod
    def StohasticNNCocktail(size):
        initPop = []
        index = 0
        while index < 5 and index < size:
            initPop.append(Solution(permutation = InitHeuristics.stohasticNN(0.75)))
            index+=1
            
        while index < 20 and index < size:
            initPop.append(Solution(permutation = InitHeuristics.coinFlipNN(random())))
            index+=1
              
        while index < 30 and index < size:
            initPop.append(Solution(permutation = InitHeuristics.coinFlipNN(random()/2)))
            index+=1
            
        while index < 35 and index < size:
            initPop.append(Solution(permutation = InitHeuristics.stohasticNN(random())))
            index+=1
            
        while index < size:
            initPop.append(Solution(permutation = InitHeuristics.monteCarloInit(50)))
            index+=1
            
        return initPop
    
    ## Linear Monte Carlo + Two Opt LSO on every solution
    @staticmethod
    def MonteCarloTwoOpt(size):
        initPop = PopulationInitializations.LinearMonteCarlo(size)
        
        for sol in initPop:
            sol = LSO.TwoOpt(sol, 20)
        
        return initPop
        
## LOCAL SEARCH OPERATORS        
class LSO:
    ## go over all solutions that are i 1-transposition-neighbourhood, DEPRECATED
    @staticmethod
    def Transposition1Neighbourhood(solution, maxiter = None):
        
        initPath = solution.getCities() 
        bestCost = solution.getObjectiveValue()
        bestPath = initPath
        before = bestCost
        iteration = 0
        
        for i in range(len(initPath)):
            for j in range(i):
                newPath = np.copy(initPath)
                newPath[i], newPath[j] = newPath[j], newPath[i]
                newCost = Distances.pathCost(newPath)
                if newCost < bestCost:
                    bestCost = newCost
                    bestPath = newPath
                iteration+=1
                if maxiter is not None and maxiter < iteration:
                    break
                
            if maxiter is not None and maxiter < iteration:
                    break
                    
        solution.cities = bestPath
        solution.computeObjectiveValue()
        #print("THING IMPROVED: " + str(before - bestCost))
        return solution
    
    ## 2-opt LSO for TSP, with iteration limit, and random start when selecting first edge (so that when performing a lot of short 2-opts they don't all target the same edge)
    @staticmethod
    def TwoOpt(solution, maxIter = 100000):
        
        improved = True
        initPath = solution.getCities()
        bestCost = solution.getObjectiveValue()
        bestPath = initPath
        before = bestCost
        iteration = 0
        
        while improved or iteration <= maxIter:
            improved = False
            randomStart = randrange(0, len(initPath)-3)
            for i in range(randomStart, len(initPath)-2):
                for j in range(i+2, len(initPath)):
                    iteration+=1
                    newPath = np.copy(bestPath)
                    newPath[i:j] = np.flip(newPath[i:j])
                    newCost = Distances.pathCost(newPath)
                    if newCost < bestCost:
                        improved = True
                        bestCost = newCost
                        bestPath = newPath
                        break
                    
                    if iteration > maxIter:
                        break
                        
                if improved or iteration > maxIter:
                    break
                
        solution.cities = bestPath
        solution.objectiveValue = bestCost
        
        #print("THING IMPROVED: " + str(before - bestCost))
        return solution


## SELECTIONS
## class that encapsulates various selection operators
class Selections:
    
    fitnessWeights = None
    rankingWeights = None
    
    # Round-Robin tournament selection function
    ## added possiblity of skipping consecutive near-duplicates to promote diversity (used after 300 iterations)
    @staticmethod
    def roundRobin(population, numberOfSelected, numberOfTrials, skipConsecutiveClones = False):
        testedSolutions = []
        numCities = len(population[0].cities)
        for sol in population:
            tempScore = 0
            opponents = np.random.choice(population, numberOfTrials, replace=False)
            for opponent in opponents:
                if sol.getObjectiveValue() > opponent.getObjectiveValue():
                    tempScore+=1
                    
            testedSolutions.append((sol, tempScore))
        
        sortedTuples = sorted(testedSolutions, key = lambda x:x[1])
        
        newContenders = [x[0] for x in sortedTuples]
        newPopulation = []
        
        if skipConsecutiveClones:
            #print("HD: ", end = "")
            canRemove = True
            i = 0
            cc = 0
            while i  < numberOfSelected:
                
                if canRemove and i > 0 and Distances.hammingDistance(newPopulation[-1].cities, newContenders[cc].cities) < 2:
                    canRemove = False
                else:
                    newPopulation.append(newContenders[cc])
                    canRemove = True
                    i+=1
                cc+=1
        else:
            newPopulation = newContenders[:numberOfSelected]
        #print(removed)
        return newPopulation
    
    
    @staticmethod
    def UpdateFitnessWeights(population):
        Selections.fitnessWeights = [1/x.getObjectiveValue() for x in population]
    
    @staticmethod
    def SimpleFitnessSelection(population, numberOfSelected):
        return choices(population, weights = Selections.fitnessWeights, k = numberOfSelected)
    
    
    @staticmethod
    def UpdateRankingWeights(population):
        population.sort()
        num = len(population)
        Selections.rankingWeights = [num-i for i in range(num)]
    
    @staticmethod
    def RankingSelection(population, numberOfSelected):
        return choices(population, weights = Selections.rankingWeights, k=  numberOfSelected)

    
    @staticmethod
    def kTournament(population, numberOfSelected, k = 3):
        selected = []
        for i in range(numberOfSelected):
            elements = np.random.choice(population, k, replace=False)
            selected.append(min(elements, key = lambda x: Diversity.sharedFitness(x, population)))
            
        return selected

## CROSSOVER
## class that holds crossover functions        
class Crossovers:
    
    ## simple crossover that randomly splits the solutions and takes care that no city appear 2 times
    @staticmethod
    def crossoverSimple(solution1, solution2):
        cities1 = solution1.getCities()
        cities2 = solution2.getCities()
        length = len(cities1)
        
        childCities1 = np.zeros((length, ), dtype = int)
        childCities2 = np.zeros((length, ), dtype = int)
        #print(childCities)
        
        randomIndex1 = randrange(0, length)
        randomIndex2 = randrange(randomIndex1, length)
        
        childIndex = 0
        usedCities1 = set()
        usedCities2 = set()
        for i in range(randomIndex1, randomIndex2+1):
            childCities1[childIndex] = cities1[i]
            childCities2[childIndex] = cities2[i]
            childIndex+=1
            usedCities1.add(cities1[i])
            usedCities2.add(cities2[i])
            
        childIndexTemp = childIndex    
        for city in cities2:
            if city not in usedCities1:
                childCities1[childIndex] = city
                childIndex+=1
                
        childIndex = childIndexTemp
        
        for city in cities1:
            if city not in usedCities2:
                childCities2[childIndex] = city
                childIndex+=1
            
        #print(childCities)
        newAlpha = SolutionParameters.parameterCrossover(solution1.alpha, solution2.alpha)
        newCrossoverProb = SolutionParameters.parameterCrossover(solution1.crossoverProb, solution2.crossoverProb)
        childCities = min([childCities1, childCities2], key = lambda x: Distances.pathCost(x))
        child = Solution(permutation = childCities, alpha = newAlpha, crossoverProb = newCrossoverProb)
        
        return child
    
    ## SEQUENTIAL CONSTRUCTIVE CROSSOVER
    @staticmethod
    def crossoverSCX(solution1, solution2):
        
        cities = [solution1.getCities(), solution2.getCities()]
        
        currentNode = 0
        takenCities = set()
        length = len(solution1.getCities())
        childCities = np.zeros((length, ), dtype = int)
        index = 0
        
        C1 = cities[0][0]
        C2 = cities[1][0]
        
        if Distances.getDistance(0, C1) < Distances.getDistance(0, C2):
            currentNode = C1
        else:
            currentNode = C2
        
        takenCities.add(currentNode)
        childCities[index] = currentNode
        index+=1
        
        while index < length:
            candidates = [-1, -1]
            for i in range(2):
                currentNodeIndex = np.where(cities[i] == currentNode)[0][0]
                if currentNodeIndex >= length-1 or (cities[i][currentNodeIndex+1] in takenCities):
                    for j in range(1, length+1):
                        if j not in takenCities:
                            candidates[i] = j
                            break
                else:
                    candidates[i] = cities[i][currentNodeIndex+1]
            
            choosenOne = min(candidates[0], candidates[1], key = lambda x:Distances.getDistance(currentNode, x))

            takenCities.add(choosenOne)
            childCities[index] = choosenOne
            currentNode = choosenOne
            index+=1
            
        newAlpha = SolutionParameters.parameterCrossover(solution1.getAlpha(), solution2.getAlpha())
        newCrossoverProb = SolutionParameters.parameterCrossover(solution1.crossoverProb, solution2.crossoverProb)
        child = Solution(permutation =  childCities, alpha = newAlpha, crossoverProb = newCrossoverProb)
        
        return child



## DIVERSITY CLASS
## encapsulates functionalities necessary for population diversity    
class Diversity:
    
    alpha = None
    sigma = None

    ## computes shared fitness objective value of given solution  for given population
    @staticmethod
    def sharedFitness(solution, population, initMultiplier = 0):
        
        multiplierPenalty = initMultiplier
        
        for otherSol in population:
            tempDist = Distances.hammingDistance(solution.cities, otherSol.cities)
            if tempDist < Diversity.sigma:
                multiplierPenalty += 1 - (tempDist / Diversity.sigma) ** Diversity.alpha 
        
        return solution.getObjectiveValue() * multiplierPenalty
    
    @staticmethod
    def setSharedFitnessParameters(alpha, sigma):
        Diversity.alpha = alpha
        Diversity.sigma = sigma
        return

# SOLUTION PARAMETERS
# Class responsible for self-adaptive parameters (initializations, crossovers etc.)
class SolutionParameters:
    
    # Alpha lower bound
    alpha_LB = None
    # Alpha upper bound
    alpha_UB = None
    
    # Lower bound for crossover
    crossover_LB = None
    #Upper bound for crossover
    crossover_UB = None
    
    ## functions to configure parameters lower and upper bounds
    @staticmethod
    def configureAlphaParameters(alpha_LB, alpha_UB):
        SolutionParameters.alpha_LB = alpha_LB
        SolutionParameters.alpha_UB = alpha_UB
    
    @staticmethod
    def configureCrossoverParameters(crossover_LB, crossover_UB):
        SolutionParameters.crossover_LB = crossover_LB
        SolutionParameters.crossover_UB = crossover_UB
        
    ## ALPHA -- get random value between bounds
    @staticmethod
    def getRndAlpha():
        return uniform(SolutionParameters.alpha_LB, SolutionParameters.alpha_UB)
    
    ## CROSSOVER PROBABILITY -- get random value between bounds
    @staticmethod
    def getRndCrossoverProb():
        return uniform(SolutionParameters.crossover_LB, SolutionParameters.crossover_UB)
    
    
    ## function for self-adaptive parameter crossover
    @staticmethod
    def parameterCrossover(param1, param2):
        #beta = 2 * random() - SolutionParameters.alpha_LB
        return uniform(min(param1, param2), max(param1, param2))
    

        
# single solution instance - a chromosome class
# has permutation of cities as atribute, and all the important methods
# self.cities --> permutation attribute --> 1D numpy array
class Solution:

    # constructor for Solution class
    # - permutation - given permutation of cities, if left out, it will generate random permutation, same with alpha and crossover probability
    def __init__(self, permutation=None, alpha = None, crossoverProb = None):
        self.mutate = self.mutateRSM
        self.cities = permutation
        self.alpha = alpha
        self.crossoverProb = None
        
        if self.alpha is None:
            self.alpha = SolutionParameters.getRndAlpha()
        
        if permutation is None:
            self.cities = Distances.randomPermutation()
            
        if self.crossoverProb is None:
            self.crossoverProb = SolutionParameters.getRndCrossoverProb()

        self.length = len(self.cities)
        self.objectiveValue = None
        self.computeObjectiveValue()

    # method to compute objective value of solution
    # stores the value so no need to compute obj. val. again
    def computeObjectiveValue(self):
        self.objectiveValue = Distances.pathCost(self.cities)
        return

    ## returns objective value of solution
    ## USE THIS TO ACCESS FITNESS
    def getObjectiveValue(self):
        if self.objectiveValue is None:
            self.computeObjectiveValue()

        return self.objectiveValue

    ## getter for alpha
    def getAlpha(self):
        return self.alpha

    # Simple swap mutation method !!! DEPRECATED !!!
    def mutateSwap(self):
        if random() < self.parameters.alpha:
            ind = np.random.choice(range(self.length), 2, replace=False)
            # print(self.cities)
            self.cities[ind[0]], self.cities[ind[1]] = self.cities[ind[1]], self.cities[ind[0]]
            # print(self.cities)

            self.computeObjectiveValue()
        return
    
    ## RSM mutation -- showed good results
    def mutateRSM(self):
        index1 = randrange(0, self.length)
        index2 = randrange(0, self.length)
        inverseStart = min(index1, index2)
        inverseEnd = max(index1,index2)
        
        self.cities[inverseStart:inverseEnd+1] = np.flip(self.cities[inverseStart:inverseEnd+1])    
        self.computeObjectiveValue()
    
        return
        
    def mutate(self):
        return



    def getCities(self):
        return self.cities
    

    # comparator so that in case of sorting solutions are sorted by fitness value
    def __lt__(self, other):
        return self.getObjectiveValue() < other.getObjectiveValue()

    # returns a string format of cities in solution
    def __str__(self):
        strVal = "[ 0 "
        for city in self.cities:
            strVal += (" " + str(city))
        strVal += " 0 ]"
        return strVal



# CLASS FOR THE TRAVELLING SALESMAN PROBLEM
class TravellingSalesman:
    def __init__(self, populationSize, offspringSize, selectionThreshold, crossover, convergeHistory, convergencethreshold):
        self.populationSize = populationSize
        #self.solutionParameters = solutionParameters
        self.offspringSize = offspringSize
        self.population = []
        self.selectionThreshold = selectionThreshold
        self.convergeHistory = convergeHistory
        self.convergencethreshold = convergencethreshold
        self.crossover = crossover
        self.lastHmeans = []

        self.lastmean = None
        self.bestSolution = None
        
        self.offspring = None
        
        self.step = 0
        
        return
    


    def initialize(self):
        ## select initalization heuristics based on how many inf edges there are
        if Distances.numberOfInfs < 20:
            ## using various different init heuristics
            self.population = PopulationInitializations.MonteCarloTwoOpt(25)
            self.population = self.population + PopulationInitializations.StohasticNNCocktail(15)
            self.population = self.population + PopulationInitializations.LinearMonteCarlo(20)
            self.population = self.population + [Solution() for x in range(60, self.populationSize)]
        
        else:
            self.population = PopulationInitializations.StohasticNNCocktail(15)
            self.population = self.population + [Solution(permutation = InitHeuristics.randomWalk()) for x in range(15, self.populationSize)]

        return

    # returns mean objective value in population
    def meanObjectiveValue(self):
        totalObjVal = sum(sol.getObjectiveValue() for sol in self.population)
        return totalObjVal / self.populationSize

    # returns best objective value in population
    def getBestSolution(self):
        #return min(self.population, key = lambda x:x.getObjectiveValue())
        return self.bestSolution


    # fast and simple k tour selection for cases with large number of cities
    def kTourSelectionSimple(self, K_size):
        elements = np.random.choice(self.population, K_size, replace = False)
        return min(elements)

    # k tour Selection that implements shared fitness
    def kTourSelectionDiversity(self, K_size):
        elements = np.random.choice(self.population, K_size, replace=False)
        return min(elements, key = lambda x: Diversity.sharedFitness(x, self.population))

    ## generate offspringSize amount of elements
    def generateOffspring(self):
        self.step += 1
        offspring = []
        selectSelection = (random() < self.selectionThreshold)
        if selectSelection:
            Selections.UpdateFitnessWeights(self.population)
        #Selections.UpdateRankingWeights(self.population)
        for i in range(self.offspringSize):
            child = None
            parent1 = None
            parent2 = None
            #parent1 = self.selection(5)
            #parent2 = self.selection(3)
            if selectSelection:
                [parent1, parent2] = Selections.SimpleFitnessSelection(self.population, 2)
            else:
                if Distances.numberOfCities() < 400:
                    parent1 = self.kTourSelectionDiversity(5)
                    parent2 = self.kTourSelectionDiversity(3)
                else:
                    parent1 = self.kTourSelectionSimple(3)
                    parent2 = self.kTourSelectionSimple(3)
                    
            combinedCrossoverProb = (parent1.crossoverProb + parent2.crossoverProb) / 2
            
            if random() < combinedCrossoverProb:

                child = self.crossover(parent1, parent2)
                child.mutate()
                
            else:
                child = Solution(permutation = InitHeuristics.monteCarloInit(10))
                child = LSO.TwoOpt(child, 10)
            #child.mutate()
            offspring.append(child)
            
        self.offspring = offspring
        return offspring

    ## ELIMINATION CONSISTING OF LSO TO A SMALL AMOUNT OF SOLUTIONS, GLOBAL MUTATION IN SOME CASES, AND ROUND ROBIN SELECTION AT THE END
    def eliminate(self):
        Alltogether = self.population + self.offspring
        bestSol = Alltogether[0]
        bestCost = Alltogether[0].getObjectiveValue()
        
        ## APPLY 2-opt LSO TO A SMALLER NUMBER OF INDIVIDUALS
        for sol in Alltogether:
            if random() < 0.05:
                sol = LSO.TwoOpt(sol, 15)                
            
            if sol.getObjectiveValue() < bestCost:
                bestCost = sol.getObjectiveValue()
                bestSol = sol
                
        self.bestSolution = bestSol
        ## global mutation probability that increases with time, in case of large number of cities, this will not happen becuase it would be too slow
        if random() < min((self.step/500)*0.2, 0.2) and Distances.numberOfCities() < 400:
            for sol in self.population:
                if sol != bestSol:
                    sol.mutate()
        
        rrSkip = False
        if self.step > 1000:
            rrSkip = True                     
            
        self.population = [bestSol] + Selections.roundRobin(Alltogether, self.populationSize-1, 10, rrSkip)
        
        return
    
    ## Elimination that combines k-tournament and crowding diversity, DEPRECATED
    def kTourAndCrowdingElimination(self):
        Alltogether = self.population + self.offspring
        bestSol = Alltogether[0]
        bestCost = Alltogether[0].getObjectiveValue()
        
        for sol in Alltogether:
            if sol.getObjectiveValue() < bestCost:
                bestCost = sol.getObjectiveValue()
                bestSol = sol
                
        self.bestSolution = bestSol

        nextGen = [bestSol]
        
        inserted = 1
        while inserted < self.populationSize:
            candidate = min(np.random.choice(Alltogether, 10, replace = False))
            testers = np.random.choice(Alltogether, 15, replace = False)
            toBeRemoved = min(testers, key = lambda x: Distances.hammingDistance(candidate.getCities(), x.getCities()))
            Alltogether.remove(toBeRemoved)
            nextGen.append(candidate)
            inserted+=1
        
        self.population = nextGen
        return

    ## method to check if algorithm is done
    def convergenceTest(self):
        ## keep track of last convergenceTest mean values and 
        if(len(self.lastHmeans) < self.convergeHistory-1):
            return False
        
        for i in range(1,len(self.lastHmeans)):
            if abs(self.lastHmeans[i-1] - self.lastHmeans[i]) > self.convergencethreshold:
                return False
                

        return True
    
    def updateMeanObjValue(self, meanObj):
        if meanObj == inf:
            return
        self.lastHmeans.append(meanObj)
        if len(self.lastHmeans) > self.convergeHistory:
            self.lastHmeans.pop(0)
        return
    
    ## apply mutation to list of solutions !! IN PLACE !!
    def mutatePopulation(self):
        for sol in self.population:
            sol.mutate()
            
        return
    
    def populationStdDev(self):
        objectives = []
        for sol in self.population:
            objectives.append(sol.getObjectiveValue())
        
        
        return stats.stdev(objectives)


# Modify the class name to match your student number.
class GeneticAlgo:


    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        Distances.setMatrix(distanceMatrix)
        print("Data loaded: number of cities " + str(Distances.numberOfCities()))
        
        ALPHA_UPPER_BOUND = 0.2
        ALPHA_LOWER_BOUND = 0.01
        CROSSOVER_LOWER_BOUND = 0.65
        CROSSOVER_UPPER_BOUND = 0.9
        ALPHA = (ALPHA_UPPER_BOUND + ALPHA_LOWER_BOUND)/2
        
        POPULATION_SIZE = 70
        OFFSPRING_SIZE = 70
        SEL_TRESHOLD = 0.3
        if Distances.numberOfCities() > 400:
            POPULATION_SIZE = 150
            OFFSPRING_SIZE = 150 
            SEL_TRESHOLD = 0 ## never do fitness selection, only simple k-tour
            
        THRESHOLD = 0.001
        HISTORY = 50
       
        SF_ALPHA = 1
        SIGMA = 0.05 * Distances.numberOfCities()
       
        Diversity.setSharedFitnessParameters(alpha = SF_ALPHA, sigma = SIGMA)        

        SolutionParameters.configureAlphaParameters(alpha_LB = ALPHA_LOWER_BOUND, alpha_UB = ALPHA_UPPER_BOUND)
        SolutionParameters.configureCrossoverParameters(crossover_LB = CROSSOVER_LOWER_BOUND, crossover_UB = CROSSOVER_UPPER_BOUND)
        
        travellingSalesman = TravellingSalesman(populationSize=POPULATION_SIZE, offspringSize=OFFSPRING_SIZE,
                                                crossover = Crossovers.crossoverSCX, selectionThreshold=SEL_TRESHOLD, convergeHistory = HISTORY, convergencethreshold=THRESHOLD)

        travellingSalesman.initialize()
        travellingSalesman.mutatePopulation()


        I = 0
        lastBest = 0
        while not travellingSalesman.convergenceTest():

            travellingSalesman.generateOffspring()
            travellingSalesman.eliminate()
            meanObjective = travellingSalesman.meanObjectiveValue()
            travellingSalesman.updateMeanObjValue(meanObjective)
            bestSolution = travellingSalesman.getBestSolution()
            bestObjective = bestSolution.getObjectiveValue()
            lastBest  = bestObjective
            
            if I % 10 == 0:
                print("Iteration: "+ str(I) + "-> Mean obj. value: " + str(meanObjective))
            I+=1

        
        print("BEST OBJ VALUE  AT THE END: " + str(lastBest) + " -- Iterations: " + str(I))
        
        
        return 0
    
    
def main():
    fileName = input("Enter file name: ")
    genAlgo = GeneticAlgo()
    genAlgo.optimize(fileName)
    return

main()
