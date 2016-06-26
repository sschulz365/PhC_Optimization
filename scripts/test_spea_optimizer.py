#Sean Billings, 2015
import random
import numpy
import subprocess
from backend import constraints
from backend.experiment import W1Experiment
from backend.objectiveFunctions import WeightedSumObjectiveFunction, IdealDifferentialObjectiveFunction
import math
from backend.spea_optimizer import SpeaOptimizer
from backend.photonicCrystalDesign import PhCWDesign
from backend.paretoFunctions import ParetoMaxFunction


# absolute path to the mpb executable
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/W1_2D_v04.ctl.txt"

# absolute path to the output ctl
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/optimizerTestFile.txt"


# we define a general experiment object
# that we reuse whenever we need to make a command-line mpb call
# see experiment.py for functionality
experiment = W1Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # accepts an int from 0 to 5
experiment.setBand(23)



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

# see constraints.py
constraintFunctions = [constraints.latticeConstraintsLD]

pcw = PhCWDesign(paramMap, 0, constraintFunctions)

#Initialize pareto function

key_map = {}
key_map["ng0"] = "max"
key_map["loss_at_ng0"] = "min"

pareto_function = ParetoMaxFunction(experiment, key_map)

#Optimizer parameters

max_generation = 10 # number of iterations of the SPEA algorithm
population_size = 10 # number of solutions to consider
pareto_archive_size = 8 # number of solutions to store after each generation
tournament_selection_rate  = 5 # number of solutions to consider in crossover/mutation



# Run the optimizer

print "Starting SPEA"

population = SpeaOptimizer.createPopulation(population_size, pcw)

optimizer = SpeaOptimizer(pareto_function)

optimizer.optimize(population,max_generation,tournament_selection_rate, pareto_archive_size)




print "\nSPEA solutions generated"

