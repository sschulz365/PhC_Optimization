#Sean Billings, 2015
import random
import numpy
import subprocess
import constraints
from experiment import Experiment
from objectiveFunctions import WeightedSumObjectiveFunction, IdealDifferentialObjectiveFunction
from waveGuideMPBOptimizer import differentialEvolution, createPopulation, gradientDescentAlgorithm
import math
from de_optimizer import DeOptimizer
from photonicCrystalDesign import PhCWDesign

paramMap = {}
paramMap["s1"] = 0 # First row vertical shift
paramMap["s2"] = 0 # Second row vertical shift
paramMap["s3"] = 0 # Third row vertical shift
#paramMap["p1"] = 0 # First row horizontal shift
#paramMap["p2"] = 0 # Second row horizontal shift
#paramMap["p3"] = 0 # Third row horizontal shift
paramMap["r0"] = 0.3 # Default air-hole radius
paramMap["r1"] = 0.3 # Default first row radius
paramMap["r2"] = 0.3 # Default second row radius
paramMap["r3"] = 0.3 # Default third row radius

# subprocess call attempts to hide the following warning
# that will be ouput in the command prompt
# for some python Development environments
#
# Some deprecated features have been used.  Set the environment
# variable GUILE_WARN_DEPRECATED to "detailed" and rerun the
# program to get more information.  Set it to "no" to suppress
# this message.

#subprocess.call("export GUILE_WARN_DEPRECATED=no", shell = True)

# absolute path to the mpb executable
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/W1_2D_v04.ctl.txt"

# absolute path to the output ctl
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/optimizerTestFile.txt"


# we define a general experiment object
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
"""
constraintFunctions = [ constraints.constraintAR1, constraints.constraintAR2,
                        constraints.constraintAR3, constraints.constraintAR0,
                        constraints.constraint0R1, constraints.constraint0R2,
                        constraints.constraint0R3, constraints.constraint0R0]
"""
# see constraints.py
constraintFunctions = [constraints.latticeConstraintsLD]


pcw = PhCWDesign(paramMap, 0, constraintFunctions)


max_generation = 10 # number of iterations of the DE alg
population_size = 20 # number of solutions to consider in DE
random_update = 0.2 # chance of updating vector fields in DE alg
elite_size = 10 # number of solutions to store in DE, and use for GD
band = 23 # band of interest for MPB computations

# specify the weights for the IdealDifferentialObjectiveFunction
"""
w1 = 0 #0.01 # bandwidth weight
w2 = 30 #100 # group index weight
w3 = 0 # average loss weight
w4 = 0 # BGP weight
w5 = 30 #0.002 # loss at ngo (group index) weight
w6 = 0
"""

# specify the weights for the WeightedSumObjectiveFunction

w1 = 0 # bandwidth weight
w2 = 50 # group index weight
w3 = 0 # average loss weight
w4 = 0 #0.6 # BGP weight
w5 = 0.02 # loss at ngo (group index) weight
w6 = 1 # delay weight

# these wights are use in the Objective Function to score mpb results
weights = [ w1, w2, w3, w4, w5, w6]

"""
ideal_group_index = 30 #self.ideal_solution[0]
ideal_bandwidth = 0.007 #self.ideal_solution[1]
ideal_loss_at_group_index = 30 #self.ideal_solution[2]
ideal_bgp = 0.3 #self.ideal_solution[3]
ideal_delay = 300 #self.ideal_solution[4]


ideal = [ideal_group_index, ideal_bandwidth, ideal_loss_at_group_index, ideal_bgp, ideal_delay]
"""

#Initialize objective function
#objFunc = IdealDifferentialObjectiveFunction(weights, experiment, ideal)
objFunc = WeightedSumObjectiveFunction(weights, experiment)

# Differential Evolution section

print "Starting Differential Evolution Optimizer"
# DEsolutions is an array of solutions generated by the DE alg

population = DeOptimizer.createPopulation(population_size, pcw)

optimizer = DeOptimizer(objFunc)

optimizer.optimize(population, max_generation, random_update, elite_size)




"""
DEsolutions = differentialEvolution(constraintFunctions, objFunc,
                                  max_generation, population_size, random_update,
                                  paramMap, elite_size, experiment)
"""
print "\nDifferential Evolution solutions generated"

