# Test Laplacian
import utilities
import constraints
import math
from waveGuideMPBOptimizer import createPopulation
from objectiveFunctions import weightedSumObjectiveFunction
from experiment import Experiment

# absolute path to the mpb executable
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/documents/W1_2D_v03.ctl.txt"

# absolute path to the output ctl
outputFile = "/Users/sean/documents/laplacianTestFile.txt"

experiment = Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # accepts an int from 0 to 5
experiment.setBand(23)

w1 = 0.05
w2 = 500
w3 = 0
w4 = 0.01
w5 = 0.002

paramMap = {}
paramMap["s1"] = 0 # First row horizontal shift
paramMap["s2"] = 0 # Second row vertical shift
paramMap["s3"] = 0 # Third row vertical shift
paramMap["p1"] = 0 # First row horizontal shift
paramMap["p2"] = 0 # Second row horizontal shift
paramMap["p3"] = 0 # Third row horizontal shift
# paramMap["r0"] = 0.3 # Default air-hole radius
paramMap["r1"] = 0.3 # Default first row radius
paramMap["r2"] = 0.3 # Default second row radius
paramMap["r3"] = 0.3 # Default third row radius

weights = [ w1, w2, w3, w4, w5]

constraintFunctions = [constraints.latticeConstraintsLD]

population_size = 3

population = createPopulation(constraintFunctions, population_size, paramMap)

print population

for solution in population:
    
    results = weightedSumObjectiveFunction(weights, solution, experiment)
    solution_score = results[0]
    bandwidth = results[1]
    group_index = results[2]
    avgLoss = results[3]
    bandwidth_group_index_product = results[4]
    loss_at_ng0 = results[5]
    print "\nSolution: " + str(solution)
    print "\nScore: " + str(solution_score)
    print "\nNormalized Bandwidth: " +  str(bandwidth)
    print "\nGroup Index: " + str(group_index)
    print "\nAverage Loss: " + str(avgLoss)
    print "\nLoss at Group Index: " + str(loss_at_ng0)
    print "\nBGP: " + str(bandwidth_group_index_product)
    
    print "\nComputing Fabrication yield"
    
    laplacian = utilities.computeLaplacian(weights, weightedSumObjectiveFunction, solution, experiment)
    fabrication_yield = 0
    for objective in laplacian.keys():
        print str(objective) + ": " + str(laplacian[objective])
        fabrication_yield = fabrication_yield + math.fabs(laplacian[objective])
        
    print "\nFabrication Yeild: " + str(fabrication_yield)

print "\nTest Complete"
