#Sean Billings, 2015

from backend import constraints
from backend.experiment import W1Experiment
from backend.objectiveFunctions import WeightedSumObjectiveFunction, IdealDifferentialObjectiveFunction
import math
from backend.spea_optimizer import SpeaOptimizer
from backend.paretoFunctions import ParetoMaxFunction
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
experiment = W1Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # accepts an int from 0 to 5
experiment.setBand(23)


# see constraints.py
constraintFunctions = [constraints.latticeConstraintsLD]

solutions = [{'r0': 0.273167, 'r1': 0.2, 'r2': 0.2, 'r3': 0.249118, 's3': 0.033635, 's2': 0.00241, 's1': 0.040613},
            {'r0': 0.244295, 'r1': 0.2, 'r2': 0.263526, 'r3': 0.237843, 's3': -0.003447, 's2': 0.004997, 's1': -0.059391},
            {'r0': 0.2, 'r1': 0.2, 'r2': 0.206285, 'r3': 0.261162, 's3': -0.006669, 's2': 0.004421, 's1': -0.045711},
            {'r0': 0.20856, 'r1': 0.22671, 'r2': 0.283094, 'r3': 0.307676, 's3': -0.005316, 's2': 0.005856, 's1': -0.070646},
            {'r0': 0.2, 'r1': 0.2, 'r2': 0.338668, 'r3': 0.228329, 's3': -0.003447, 's2': -0.005503, 's1': -0.098542},
            {'r0': 0.2, 'r1': 0.219778, 'r2': 0.224981, 'r3': 0.290696, 's3': -0.005837, 's2': -0.003223, 's1': -0.048862},
            {'r0': 0.214559, 'r1': 0.2, 'r2': 0.324682, 'r3': 0.2864, 's3': 0.002134, 's2': 0.004976, 's1': -0.057412},
            {'r0': 0.2, 'r1': 0.225582, 'r2': 0.338668, 'r3': 0.228329, 's3': -0.006717, 's2': -0.005503, 's1': -0.098542},
            {'r0': 0.2, 'r1': 0.2, 'r2': 0.283094, 'r3': 0.261162, 's3': -0.006669, 's2': 0.004421, 's1': -0.064895},
            {'r0': 0.2, 'r1': 0.222577, 'r2': 0.285996, 'r3': 0.309276, 's3': -0.003447, 's2': 0.004508, 's1': -0.051605},
            {'r0': 0.230147, 'r1': 0.2, 'r2': 0.316369, 'r3': 0.256247, 's3': -5.3e-05, 's2': -0.002631, 's1': 0.008945},
            {'r0': 0.227923, 'r1': 0.2, 'r2': 0.281751, 'r3': 0.261634, 's3': -0.006472, 's2': 0.00636, 's1': -0.037804},
            {'r0': 0.2, 'r1': 0.2, 'r2': 0.268639, 'r3': 0.290696, 's3': -0.005837, 's2': -0.001401, 's1': -0.048862},
            {'r0': 0.2, 'r1': 0.2, 'r2': 0.258431, 'r3': 0.290696, 's3': -0.006472, 's2': -0.001401, 's1': -0.048862},
            {'r0': 0.248038, 'r1': 0.2, 'r2': 0.258431, 'r3': 0.261095, 's3': 6.2e-05, 's2': -0.00416, 's1': -0.045961},
            {'r0': 0.224682, 'r1': 0.2, 'r2': 0.243644, 'r3': 0.261162, 's3': -0.003447, 's2': 0.006156, 's1': -0.084962},
            {'r0': 0.215179, 'r1': 0.2, 'r2': 0.301223, 'r3': 0.256488, 's3': -0.006669, 's2': -0.00416, 's1': -0.045909},
            {'r0': 0.2, 'r1': 0.2, 'r2': 0.283094, 'r3': 0.261162, 's3': -0.005316, 's2': 0.004093, 's1': -0.070646}
             ]

population = []
for vector in solutions:
    pcw = PhCWDesign(vector, 0, constraintFunctions)
    population.append(pcw.copy_phc)

max_generation = 15 # number of iterations for SPEA
#population_size = 10 # number of solutions to consider in SPEA
pareto_archive_size = 40 # number of solutions to store in the SPEA PAS
tournament_selection_rate  = 5 # number of solutions to consider for evolution in SPEA

#Initialize objective function
#objFunc = IdealDifferentialObjectiveFunction(weights, experiment, ideal)
key_map = {}
key_map["ng0"] = "max"
key_map["loss_at_ng0"] = "min"

pareto_function = ParetoMaxFunction(experiment, key_map)

# Differential Evolution section

print "Starting SPEA"


optimizer = SpeaOptimizer(pareto_function)

optimizer.optimize(population,max_generation,tournament_selection_rate, pareto_archive_size)
