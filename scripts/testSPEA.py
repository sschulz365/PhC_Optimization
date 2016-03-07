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


"""
population = [  {'r0': 0.21152, 'r1': 0.2, 'r2': 0.298348, 'r3': 0.261095, 's3': -0.000926, 's2': -0.0007, 's1': 0.008945},
                {'r0': 0.2, 'r1': 0.2, 'r2': 0.283094, 'r3': 0.261162, 's3': -0.006669, 's2': 0.003112, 's1': -0.058935},
                {'r0': 0.20428, 'r1': 0.20586, 'r2': 0.242626, 'r3': 0.260383, 's3': -0.005902, 's2': 0.004414, 's1': -0.077922},
                {'r0': 0.259446, 'r1': 0.2, 'r2': 0.327154, 'r3': 0.257375, 's3': -0.000926, 's2': 0.002029, 's1': 0.008945},
                {'r0': 0.2, 'r1': 0.245968, 'r2': 0.242572, 'r3': 0.330069, 's3': 0.004151, 's2': 0.002228, 's1': -0.05655},
                {'r0': 0.228091, 'r1': 0.220457, 'r2': 0.27245, 'r3': 0.248552, 's3': -0.006083, 's2': 0.004187, 's1': -0.084846},
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.239041, 'r3': 0.303714, 's3': 0.006148, 's2': 0.013251, 's1': -0.006994},
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.235619, 'r3': 0.289706, 's3': 0.004461, 's2': -0.012459, 's1': -0.006994},
                {'r0': 0.231764, 'r1': 0.2, 'r2': 0.352973, 'r3': 0.274492, 's3': -0.002391, 's2': 0.007186, 's1': -0.006395},
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.281751, 'r3': 0.310675, 's3': -0.003429, 's2': 0.008228, 's1': -0.006247}] #,

                {'r0': 0.2, 'r1': 0.209094, 'r2': 0.305454, 'r3': 0.271954, 's3': -0.003429, 's2': 0.009054, 's1': -0.006247},
                {'r0': 0.203693, 'r1': 0.218358, 'r2': 0.233882, 'r3': 0.302653, 's3': -0.003634, 's2': 0.011336, 's1': -0.006994},
                {'r0': 0.248038, 'r1': 0.2, 'r2': 0.298348, 'r3': 0.261095, 's3': 6.2e-05, 's2': -0.0007, 's1': 0.008945},
                {'r0': 0.2, 'r1': 0.2, 'r2': 0.283094, 'r3': 0.261162, 's3': -0.006669, 's2': 0.003112, 's1': -0.06576},
                {'r0': 0.2048, 'r1': 0.2, 'r2': 0.258431, 'r3': 0.261162, 's3': -0.006669, 's2': -0.00416, 's1': -0.045961},
                {'r0': 0.2, 'r1': 0.202976, 'r2': 0.213677, 'r3': 0.266906, 's3': -0.005008, 's2': -0.004855, 's1': -0.067445},
                {'r0': 0.20428, 'r1': 0.20025, 'r2': 0.242626, 'r3': 0.262158, 's3': -0.005902, 's2': 0.006382, 's1': -0.055225},
                {'r0': 0.2, 'r1': 0.2, 'r2': 0.307305, 'r3': 0.257375, 's3': -0.000926, 's2': 0.002029, 's1': 0.008945},
                {'r0': 0.2, 'r1': 0.2, 'r2': 0.327154, 'r3': 0.285903, 's3': -0.000926, 's2': 0.002029, 's1': 0.008945},
                {'r0': 0.2, 'r1': 0.2, 'r2': 0.283094, 'r3': 0.261162, 's3': -0.005316, 's2': 0.004093, 's1': -0.070646},
                {'r0': 0.241271, 'r1': 0.212013, 'r2': 0.200502, 'r3': 0.285701, 's3': -0.029809, 's2': -0.033444, 's1': 0.020306},
                {'r0': 0.241271, 'r1': 0.2, 'r2': 0.2, 'r3': 0.269539, 's3': -0.027553, 's2': 0.038322, 's1': 0.019723},
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.251445, 'r3': 0.288757, 's3': -0.005184, 's2': 0.010405, 's1': -0.005997},
                {'r0': 0.241271, 'r1': 0.212013, 'r2': 0.2, 'r3': 0.285701, 's3': -0.029187, 's2': -0.039733, 's1': 0.020306},
                {'r0': 0.203693, 'r1': 0.218358, 'r2': 0.305454, 'r3': 0.310675, 's3': -0.005184, 's2': 0.009054, 's1': -0.006994},
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.305454, 'r3': 0.310675, 's3': -0.005184, 's2': 0.009054, 's1': -0.006224}
                ]

"""
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



"""
#population.extend(good_solutions)
print "Starting Strength Pareto Evolutionary Algorithm"


# spea_solutions is an array of solutions generated by the SPEA alg
population = strength_pareto_evolutionary_algorithm(population,
                                                       experiment,
                                                       constraintFunctions,
                                                       max_generation,
                                                       pareto_archive_size,
                                                       tournament_selection_rate)



out_stream = open(solution_file, 'a')
out_stream.write("SPEA")
out_stream.write("\n")

out_stream.close()


print "\n\n\nResults"

out_stream = open(solution_file, 'a')

for solution in population:


    results = objFunc.evaluate(solution)

    out_stream.write(str([results, solution]))
    out_stream.write("\n")


# specify the weights for the IdealDifferentialObjectiveFunction

w1 = 0 #0.01 # bandwidth weight
w2 = 40 #100 # group index weight
w3 = 0 # average loss weight
w4 = 0 # BGP weight
w5 = 40 #0.002 # loss at ngo (group index) weight
w6 = 0

# these wights are use in the Objective Function to score mpb results
weights = [ w1, w2, w3, w4, w5, w6]

objFunc = WeightedSumObjectiveFunction(weights, experiment)

ideal_group_index = 30 #self.ideal_solution[0]
ideal_bandwidth = 0.007 #self.ideal_solution[1]
ideal_loss_at_group_index = 30 #self.ideal_solution[2]
ideal_bgp = 0.3 #self.ideal_solution[3]
ideal_delay = 300 #self.ideal_solution[4]


ideal = [ideal_group_index, ideal_bandwidth, ideal_loss_at_group_index, ideal_bgp, ideal_delay]

#Initialize objective function
#objFunc = IdealDifferentialObjectiveFunction(weights, experiment, ideal)


print "Starting Gradient Descent"
descent_scaler = 0.2
completion_scaler = 0.1
alpha_scaler = 0.9



population = gradientDescentAlgorithm(objFunc,
                                     constraintFunctions,
                                     population, descent_scaler,
                                     completion_scaler, alpha_scaler)


"""
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
