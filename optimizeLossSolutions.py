#Sean Billings, 2015
import random
import numpy
import subprocess
import constraints
from experiment import Experiment
from objectiveFunctions import WeightedSumObjectiveFunction, IdealDifferentialObjectiveFunction
from waveGuideMPBOptimizer import differentialEvolution, createPopulation, gradientDescentAlgorithm
import math
from spea_optimizer import SpeaOptimizer
from photonicCrystalDesign import PhCWDesign
from paretoFunctions import ParetoMaxFunction
from photonicCrystalDesign import PhCWDesign


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
"""
solutions = [  {'r0': 0.21152, 'r1': 0.2, 'r2': 0.298348, 'r3': 0.261095, 's3': -0.000926, 's2': -0.0007, 's1': 0.008945},
                {'r0': 0.2, 'r1': 0.2, 'r2': 0.283094, 'r3': 0.261162, 's3': -0.006669, 's2': 0.003112, 's1': -0.058935},
                {'r0': 0.20428, 'r1': 0.20586, 'r2': 0.242626, 'r3': 0.260383, 's3': -0.005902, 's2': 0.004414, 's1': -0.077922},
                {'r0': 0.259446, 'r1': 0.2, 'r2': 0.327154, 'r3': 0.257375, 's3': -0.000926, 's2': 0.002029, 's1': 0.008945},
                {'r0': 0.2, 'r1': 0.245968, 'r2': 0.242572, 'r3': 0.330069, 's3': 0.004151, 's2': 0.002228, 's1': -0.05655},
                {'r0': 0.228091, 'r1': 0.220457, 'r2': 0.27245, 'r3': 0.248552, 's3': -0.006083, 's2': 0.004187, 's1': -0.084846},
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.239041, 'r3': 0.303714, 's3': 0.006148, 's2': 0.013251, 's1': -0.006994},
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.235619, 'r3': 0.289706, 's3': 0.004461, 's2': -0.012459, 's1': -0.006994},
                {'r0': 0.231764, 'r1': 0.2, 'r2': 0.352973, 'r3': 0.274492, 's3': -0.002391, 's2': 0.007186, 's1': -0.006395},
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.281751, 'r3': 0.310675, 's3': -0.003429, 's2': 0.008228, 's1': -0.006247},
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
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.305454, 'r3': 0.310675, 's3': -0.005184, 's2': 0.009054, 's1': -0.006224},
                {'r0': 0.2, 'r1': 0.222577, 'r2': 0.267186, 'r3': 0.261162, 's3': -0.003447, 's2': 0.004093, 's1': -0.070646},
                {'r0': 0.2, 'r1': 0.2, 'r2': 0.283094, 'r3': 0.261162, 's3': -0.006669, 's2': 0.004016, 's1': -0.069405},
                {'r0': 0.228091, 'r1': 0.220457, 'r2': 0.27245, 'r3': 0.248552, 's3': -0.006083, 's2': 0.004187, 's1': -0.084846},
                {'r0': 0.2048, 'r1': 0.2, 'r2': 0.258431, 'r3': 0.261162, 's3': -0.006669, 's2': -0.00416, 's1': -0.045961}
                ]
"""
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
    population.append(pcw.copy_phc())

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
