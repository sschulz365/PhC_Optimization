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
from gd_optimizer import GradientDescentOptimizer
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

population_size = 5
max_iterations = 5 # number of iterations of the DE alg
descent_scaler = 0.2
completion_scaler = 0.1
alpha_scaler = 0.9
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

"""
ideal_group_index = 30 #self.ideal_solution[0]
ideal_bandwidth = 0.007 #self.ideal_solution[1]
ideal_loss_at_group_index = 30 #self.ideal_solution[2]
ideal_bgp = 0.3 #self.ideal_solution[3]
ideal_delay = 300 #self.ideal_solution[4]


ideal = [ideal_group_index, ideal_bandwidth, ideal_loss_at_group_index, ideal_bgp, ideal_delay]

objFunc = IdealDifferentialObjectiveFunction(weights, experiment, ideal)
"""

# specify the weights for the WeightedSumObjectiveFunction

w1 = 0 # bandwidth weight
w2 = 0 # group index weight
w3 = 0 # average loss weight
w4 = 1 #0.6 # BGP weight
w5 = 0.0 # loss at ngo (group index) weight
w6 = 0 # delay weight

# these wights are use in the Objective Function to score mpb results
weights = [ w1, w2, w3, w4, w5, w6]



#Initialize objective function

objFunc = WeightedSumObjectiveFunction(weights, experiment)

# Gradient Descent section

print "Starting Gradient Descent Optimizer"




vectors = [{'p2': 0.305666, 'p3': -0.001227, 'p1': 0.226132, 'r0': 0.206906, 'r1': 0.2, 'r2': 0.277454, 'r3': 0.227293, 's3': 0.02537, 's2': 0.161428, 's1': 0.198186},
                {'p2': 0.007214, 'p3': 0.005759, 'p1': -0.13563, 'r0': 0.232891, 'r1': 0.200274, 'r2': 0.383745, 'r3': 0.2, 's3': -0.065504, 's2': -0.075669, 's1': -0.069502},
                {'p2': 0.178779, 'p3': 0.030888, 'p1': 0.124599, 'r0': 0.2279, 'r1': 0.237683, 'r2': 0.4, 'r3': 0.2, 's3': 0.076104, 's2': -0.097843, 's1': 0.063663},
                {'p2': -0.125797, 'p3': -0.050961, 'p1': 0.131258, 'r0': 0.2, 'r1': 0.2, 'r2': 0.280327, 'r3': 0.259032, 's3': -0.053, 's2': -0.141022, 's1': -0.151103},
                {'p2': 0.005009, 'p3': 0.004437, 'p1': -0.13563, 'r0': 0.232891, 'r1': 0.2, 'r2': 0.383745, 'r3': 0.2, 's3': -0.065504, 's2': -0.075669, 's1': -0.069502},
                {'p2': 0.211272, 'p3': -0.042682, 'p1': 0.124599, 'r0': 0.2, 'r1': 0.2, 'r2': 0.310876, 'r3': 0.2, 's3': -0.078033, 's2': -0.095436, 's1': -0.13246},
                {'p2': 0.005009, 'p3': 0.004437, 'p1': -0.141671, 'r0': 0.237322, 'r1': 0.203036, 'r2': 0.383745, 'r3': 0.2, 's3': -0.046694, 's2': -0.101481, 's1': -0.062471},
                {'p2': 0.005009, 'p3': 0.006596, 'p1': -0.135418, 'r0': 0.237322, 'r1': 0.2, 'r2': 0.383745, 'r3': 0.2, 's3': -0.046694, 's2': -0.094858, 's1': -0.062471},
                {'p2': 0.169217, 'p3': 0.030888, 'p1': 0.124599, 'r0': 0.2279, 'r1': 0.237683, 'r2': 0.4, 'r3': 0.2, 's3': 0.094071, 's2': -0.097843, 's1': 0.063663},
                {'p2': 0.005009, 'p3': 0.006596, 'p1': -0.115785, 'r0': 0.237322, 'r1': 0.2, 'r2': 0.383745, 'r3': 0.2, 's3': -0.046694, 's2': -0.101481, 's1': -0.062471},
                {'p2': -0.183314, 'p3': -0.050542, 'p1': -0.141671, 'r0': 0.2, 'r1': 0.260604, 'r2': 0.355695, 'r3': 0.2, 's3': 0.080771, 's2': -0.092188, 's1': -0.134903},
                {'p2': -0.209395, 'p3': 7.5e-05, 'p1': -0.152738, 'r0': 0.2, 'r1': 0.203047, 'r2': 0.276094, 'r3': 0.213233, 's3': -0.020123, 's2': -0.109988, 's1': -0.108935},
                {'p2': -0.210475, 'p3': -0.042662, 'p1': -0.162253, 'r0': 0.2, 'r1': 0.2, 'r2': 0.264382, 'r3': 0.231593, 's3': -0.070369, 's2': -0.153377, 's1': -0.12171}]

population = GradientDescentOptimizer.createPopulation(len(vectors), pcw)
i = 0
for pc in population:
    pc.solution_vector = vectors[i]
    i += 1

optimizer = GradientDescentOptimizer(objFunc)

results = optimizer.optimize(population, descent_scaler, completion_scaler, alpha_scaler, max_iterations)



print "\nGradient Descent solutions generated"

i = 0
for opt_pcw in results:
    print "Solution: " + str(i) + "\n"
    print str(opt_pcw.solution_vector) + "\n"
    print str(opt_pcw.figures_of_merit) + "\n"
    i += 1