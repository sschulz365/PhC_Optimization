#test strength_pareto_evolutionary_algorithm


import random
import numpy
import subprocess
import constraints
from experiment import Experiment
from objectiveFunctions import WeightedSumObjectiveFunction, IdealDifferentialObjectiveFunction
from waveGuideMPBOptimizer import differentialEvolution, strength_pareto_evolutionary_algorithm, createPopulation, gradientDescentAlgorithm
import utilities
import math

paramMap = {}
paramMap["s1"] = -0.117 # First row vertical shift
paramMap["s2"] = 0.039 # Second row vertical shift
paramMap["s3"] = 0 # Third row vertical shift
#paramMap["p1"] = 0 # First row horizontal shift
#paramMap["p2"] = 0 # Second row horizontal shift
#paramMap["p3"] = 0 # Third row horizontal shift
paramMap["r0"] = 0.27 # Default air-hole radius
paramMap["r1"] = 0.27 # Default first row radius
paramMap["r2"] = 0.27 # Default second row radius
paramMap["r3"] = 0.27 # Default third row radius


# absolute path to the mpb executable
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt"

# absolute path to the output ctl
outputFile = "/Users/sean/documents/optimizerTestFile.txt"

# solution record file
solution_file = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/results/solution_file.txt"

# we define a generalized experiment object
# that we reuse whenever we need to make a command-line mpb call
# see experiment.py for functionality
experiment = Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # accepts an int from 0 to 5
experiment.setBand(23)


# descriptions for these constraitns are available in the constraints.py file
#constraintFunctions = [ constraints.constraintAP1, constraints.constraint0P1,
#                constraints.constraintAP2, constraints.constraint0P2,
#                constraints.constraintAP3, constraints.constraint0P3,
#                constraints.constraintAS1, constraints.constraint0S1,
#                constraints.constraintAS2, constraints.constraint0S2,
#                constraints.constraintAR1, constraints.constraintAR2,
#                constraints.constraintAR3]

# see constraints.py
constraintFunctions = [constraints.latticeConstraintsLD]

max_generation = 3 # number of iterations of the SpeA alg
population_size = 20 # number of solutions to consider in SPEA
random_update = 0.3 # chance of updating vector fields in SPEA
pareto_archive_size = 10 # number of solutions to store in DE, and use for GD
band = 23 # band of interest for MPB computations
tournament_selection_rate = 5



# Differential Evolution section
#population.extend(createPopulation(constraintFunctions, population_size, paramMap))

population = createPopulation(constraintFunctions, population_size, paramMap)
# specify the weights for the weightedSumObjectiveFunction


w1 = 0 #0.01 # bandwidth weight
w2 = 50 #100 # group index weight
w3 = 0 # average loss weight
w4 = 0 # BGP weight
w5 = 0.02 # loss at ngo (group index) weight
w6 = 0

# these wights are use in the Objective Function to score mpb results
weights = [ w1, w2, w3, w4, w5, w6]

objFunc = WeightedSumObjectiveFunction(weights, experiment)


out_stream = open(solution_file, 'a')
out_stream.write("SPEA+RS")
out_stream.write("\n")

out_stream.close()


print "\n\n\nResults"

out_stream = open(solution_file, 'a')

for solution in population:

    
    results = objFunc.evaluate(solution)
    
    out_stream.write(str([results, solution]))
    out_stream.write("\n")
                 
                     
    solution_score = results[0]
    bandwidth = results[1]
    group_index = results[2]
    avgLoss = results[3] # average loss
    bandwidth_group_index_product = results[4] #BGP
    loss_at_ng0 = results[5] # loss at group index
    delay = results[6]
    print "\nSolution: " + str(solution)
    print "\nScore: " + str(solution_score)
    print "\nNormalized Bandwidth: " +  str(bandwidth)
    print "\nGroup Index: " + str(-group_index)
    print "\nAverage Loss: " + str(avgLoss)
    print "\nLoss at Group Index: " + str(loss_at_ng0)
    print "\nBGP: " + str(bandwidth_group_index_product)
    print "\nDelay: " + str(delay) + "\n"
    
    #print "\nComputing Fabrication Stability..."

    # optional fabrication stability computation
    """
    laplacian = utilities.computeLaplacian(weights, weightedSumObjectiveFunction, solution, experiment)
    fabrication_stability = 0
    for key in laplacian.keys():
        fabrication_stability = fabrication_stability + laplacian[key]**2

    fabrication_stability = math.sqrt(fabrication_stability)
    print "\nFabrication Stability " + str(fabrication_stability)
    """

print "\nOptimization Complete"

out_stream.close()
