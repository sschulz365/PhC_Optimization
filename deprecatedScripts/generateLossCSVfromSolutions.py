import random
import numpy
import subprocess
from backend import constraints
from backend.experiment import W1Experiment
import math
from backend import mpbParser


#Loss Solutions

population = [  {'r0': 0.21152, 'r1': 0.2, 'r2': 0.298348, 'r3': 0.261095, 's3': -0.000926, 's2': -0.0007, 's1': 0.008945},
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
                {'r0': 0.2, 'r1': 0.218358, 'r2': 0.305454, 'r3': 0.310675, 's3': -0.005184, 's2': 0.009054, 's1': -0.006224}
                ]


# absolute path to the mpb executable
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt"

# absolute path to the output ctl
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/csvLossTestFile.txt"

csv_file = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/loss_final_solutions.txt"

# we define a generalized experiment object
# that we reuse whenever we need to make a command-line mpb call
# see experiment.py for functionality
experiment = W1Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # accepts an int from 0 to 5
experiment.setBand(23)


out_stream = open(csv_file, 'w')
out_stream.write("Loss, Average Loss, Group Index, Average Group Index, r0, r1, r2, r3, s1, s2, s3")
out_stream.write("\n")


print "\n\n\nResults"



for solution in population:

    print solution

    experiment.setParams(solution)
    experiment.perform()
    results = mpbParser.parseObjFunctionParams(experiment)
    bandwidth = results[0]
    group_index = results[1]
    avgLoss = results[2] # average loss
    bandwidth_group_index_product = math.fabs(results[3]) #BGP
    loss_at_ng0 = results[4] # loss at group index
    delay = results[5]

    print "\nNormalized Bandwidth: " +  str(bandwidth)
    print "Group Index: " + str(group_index)
    print "Average Loss: " + str(avgLoss)
    print "Loss at Group Index: " + str(loss_at_ng0)
    print "BGP: " + str(bandwidth_group_index_product)
    print "Delay: " + str(delay) + "\n"

    out_string = str(loss_at_ng0) + ", " + str(avgLoss) + ", " + str(math.fabs(group_index)) + ", " + str(bandwidth_group_index_product/bandwidth) + ", "
    out_string = out_string + str(solution["r0"]) + ", " + str(solution["r1"]) + ", " + str(solution["r2"]) +  ", " + str(solution["r3"]) + ", "
    out_string = out_string + str(solution["s1"]) + ", " + str(solution["s2"]) + ", " + str(solution["s3"]) + "\n"

    out_stream.write(out_string)

print "CSV generated"

out_stream.close()
