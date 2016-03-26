#Sean Billings, 2015


from backend import constraints
from backend.experiment import Experiment
from backend.objectiveFunctions import WeightedSumObjectiveFunction, IdealDifferentialObjectiveFunction
from backend.spea_optimizer import SpeaOptimizer
from backend.paretoFunctions import ParetoMaxFunction
from backend.gd_optimizer import GradientDescentOptimizer
from backend.photonicCrystalDesign import PhCWDesign


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


# see constraints.py
constraintFunctions = [constraints.latticeConstraintsLD]

solutions = [{'p2': 0.005009, 'p3': 0.006596, 'p1': -0.135418, 'r0': 0.237322, 'r1': 0.2, 'r2': 0.383745, 'r3': 0.2, 's3': -0.046694, 's2': -0.094858, 's1': -0.062471},
 {'p2': -0.19244, 'p3': 0.001644, 'p1': -0.152738, 'r0': 0.2, 'r1': 0.266403, 'r2': 0.240238, 'r3': 0.265909, 's3': -0.019961, 's2': -0.09365, 's1': -0.131561},
 {'p2': -0.158791, 'p3': 0.030778, 'p1': -0.141671, 'r0': 0.297091, 'r1': 0.2, 'r2': 0.341488, 'r3': 0.238264, 's3': 0.05995, 's2': -0.149409, 's1': -0.095711},
 {'p2': -0.259401, 'p3': -0.000774, 'p1': 0.124599, 'r0': 0.223115, 'r1': 0.244706, 'r2': 0.292582, 'r3': 0.2, 's3': -0.07421, 's2': -0.092381, 's1': -0.123115},
 {'p2': 0.171362, 'p3': 0.001644, 'p1': 0.124599, 'r0': 0.2, 'r1': 0.2, 'r2': 0.2, 'r3': 0.2, 's3': -0.018858, 's2': -0.204392, 's1': -0.108935},
 {'p2': -0.157476, 'p3': -0.050961, 'p1': -0.145734, 'r0': 0.2, 'r1': 0.227629, 'r2': 0.28966, 'r3': 0.272386, 's3': -0.07051, 's2': -0.116947, 's1': -0.131717},
 {'p2': 0.169217, 'p3': 0.030888, 'p1': 0.124599, 'r0': 0.2279, 'r1': 0.237683, 'r2': 0.4, 'r3': 0.2, 's3': 0.094071, 's2': -0.097843, 's1': 0.063663},
 {'p2': 0.211272, 'p3': -0.042682, 'p1': 0.124599, 'r0': 0.206353, 'r1': 0.2, 'r2': 0.292582, 'r3': 0.2, 's3': -0.07421, 's2': -0.095436, 's1': -0.123115},
 {'p2': 0.126122, 'p3': -0.027806, 'p1': 0.124599, 'r0': 0.235128, 'r1': 0.242938, 'r2': 0.2, 'r3': 0.278004, 's3': 0.084515, 's2': -0.048079, 's1': 0.067113},
 {'p2': 0.200715, 'p3': 0.05068, 'p1': 0.067357, 'r0': 0.2279, 'r1': 0.237683, 'r2': 0.4, 'r3': 0.2, 's3': -0.091657, 's2': -0.077118, 's1': 0.063663},
 {'p2': 0.005009, 'p3': 0.006596, 'p1': -0.115785, 'r0': 0.237322, 'r1': 0.2, 'r2': 0.383745, 'r3': 0.2, 's3': -0.046694, 's2': -0.101481, 's1': -0.062471},
 {'p2': 0.234399, 'p3': -0.001729, 'p1': 0.124599, 'r0': 0.2, 'r1': 0.2, 'r2': 0.310876, 'r3': 0.2, 's3': -0.078033, 's2': -0.095436, 's1': -0.13246},
 {'p2': 0.005009, 'p3': 0.004437, 'p1': -0.141671, 'r0': 0.220114, 'r1': 0.203036, 'r2': 0.383745, 'r3': 0.2, 's3': -0.046694, 's2': -0.101481, 's1': -0.069293},
 {'p2': 0.269865, 'p3': -0.002661, 'p1': -0.141671, 'r0': 0.223115, 'r1': 0.231531, 'r2': 0.383745, 'r3': 0.2, 's3': -0.040287, 's2': -0.101481, 's1': 0.058},
 {'p2': -0.209395, 'p3': 0.001035, 'p1': -0.152738, 'r0': 0.266314, 'r1': 0.2, 'r2': 0.203594, 'r3': 0.213233, 's3': -0.020123, 's2': -0.095436, 's1': 0.063663},
 {'p2': 0.145875, 'p3': -0.052244, 'p1': 0.117367, 'r0': 0.2, 'r1': 0.260604, 'r2': 0.4, 'r3': 0.2, 's3': 0.059225, 's2': -0.085618, 's1': -0.134903},
 {'p2': -0.209395, 'p3': 0.001035, 'p1': -0.152738, 'r0': 0.2, 'r1': 0.2, 'r2': 0.203594, 'r3': 0.213233, 's3': -0.020123, 's2': -0.109988, 's1': -0.108935},
 {'p2': 0.00519, 'p3': 0.004437, 'p1': -0.141671, 'r0': 0.223115, 'r1': 0.231531, 'r2': 0.383745, 'r3': 0.2, 's3': -0.040287, 's2': -0.101481, 's1': -0.07594},
 {'p2': 0.211272, 'p3': -0.042682, 'p1': 0.124599, 'r0': 0.2, 'r1': 0.2, 'r2': 0.310876, 'r3': 0.2, 's3': -0.078033, 's2': -0.095436, 's1': -0.13246},
 {'p2': -0.175329, 'p3': 0.001427, 'p1': 0.172176, 'r0': 0.2, 'r1': 0.250731, 'r2': 0.286036, 'r3': 0.2, 's3': -0.040287, 's2': -0.206823, 's1': -0.136598},
 {'p2': -0.183314, 'p3': -0.050542, 'p1': -0.141671, 'r0': 0.2, 'r1': 0.260604, 'r2': 0.355695, 'r3': 0.2, 's3': 0.080771, 's2': -0.092188, 's1': -0.134903},
 {'p2': 0.305666, 'p3': -0.002079, 'p1': 0.226132, 'r0': 0.2, 'r1': 0.2, 'r2': 0.244564, 'r3': 0.227293, 's3': 0.032375, 's2': 0.161428, 's1': 0.198186},
 {'p2': 0.223957, 'p3': 0.05068, 'p1': 0.124599, 'r0': 0.2279, 'r1': 0.237683, 'r2': 0.4, 'r3': 0.2, 's3': -0.091657, 's2': -0.077118, 's1': 0.063663},
 {'p2': -0.243149, 'p3': 0.000558, 'p1': 0.172176, 'r0': 0.2, 'r1': 0.250731, 'r2': 0.286036, 'r3': 0.245116, 's3': -0.040287, 's2': -0.1803, 's1': -0.136598},
 {'p2': -0.158791, 'p3': -0.050961, 'p1': 0.124599, 'r0': 0.2, 'r1': 0.227629, 'r2': 0.338433, 'r3': 0.272386, 's3': -0.07051, 's2': -0.141022, 's1': -0.131717},
 {'p2': -0.210475, 'p3': -0.050961, 'p1': 0.124599, 'r0': 0.2, 'r1': 0.227629, 'r2': 0.294188, 'r3': 0.272386, 's3': -0.07051, 's2': -0.16495, 's1': -0.13535},
 {'p2': 0.223957, 'p3': -0.042682, 'p1': -0.165299, 'r0': 0.266314, 'r1': 0.208826, 'r2': 0.335955, 'r3': 0.204147, 's3': -0.092404, 's2': -0.095436, 's1': 0.063663},
 {'p2': -0.209395, 'p3': 7.5e-05, 'p1': -0.152738, 'r0': 0.2, 'r1': 0.203047, 'r2': 0.276094, 'r3': 0.213233, 's3': -0.020123, 's2': -0.109988, 's1': -0.108935},
 {'p2': 0.211272, 'p3': -0.042682, 'p1': -0.153843, 'r0': 0.236084, 'r1': 0.2, 'r2': 0.310876, 'r3': 0.2, 's3': -0.078033, 's2': -0.095436, 's1': 0.058},
 {'p2': -0.210475, 'p3': -0.042662, 'p1': -0.162253, 'r0': 0.2, 'r1': 0.2, 'r2': 0.264382, 'r3': 0.231593, 's3': -0.070369, 's2': -0.153377, 's1': -0.12171}]


population = []
for vector in solutions:
    pcw = PhCWDesign(vector, 0, constraintFunctions)
    population.append(pcw.copy_phc())

max_generation = 20 # number of iterations for SPEA
population_size = 30 # number of solutions to consider in SPEA
pareto_archive_size = 40 # number of solutions to store in the SPEA PAS
tournament_selection_rate  = 5 # number of solutions to consider for evolution in SPEA

#Initialize objective function
#objFunc = IdealDifferentialObjectiveFunction(weights, experiment, ideal)
key_map = {}
key_map["ng0"] = "max"
key_map["bandwidth"] = "max"
key_map["GBP"] = "max"

pareto_function = ParetoMaxFunction(experiment, key_map)
"""
population2 = SpeaOptimizer.createPopulation(population_size - len(population),pcw)
for p in population2:
    population.append(p)
"""
# SPEA section

print "Starting SPEA"


optimizer = SpeaOptimizer(pareto_function)

optimizer.optimize(population,max_generation,tournament_selection_rate, pareto_archive_size)


max_iterations = 5 # number of iterations of the DE alg
descent_scaler = 0.2
completion_scaler = 0.1
alpha_scaler = 0.9


# specify the weights for the WeightedSumObjectiveFunction

w1 = 0 # bandwidth weight
w2 = 0 # group index weight
w3 = 0 # average loss weight
w4 = 1 #0.6 # GBP weight
w5 = 0.0 # loss at ngo (group index) weight
w6 = 0 # delay weight

# these wights are use in the Objective Function to score mpb results
weights = [ w1, w2, w3, w4, w5, w6]



#Initialize objective function

objFunc = WeightedSumObjectiveFunction(weights, experiment)

# Gradient Descent section

print "Starting Gradient Descent Optimizer"



optimizer2 = GradientDescentOptimizer(objFunc)

results = optimizer2.optimize(population, descent_scaler, completion_scaler, alpha_scaler, max_iterations)



for pcw in results:
    print pcw.solution_vector
    print pcw.figures_of_merit

print "\nGradient Descent solutions generated"
