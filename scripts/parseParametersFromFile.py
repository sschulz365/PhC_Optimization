from experiment import Experiment
from mpbParser import parseObjFunctionParams3D
import math

# absolute path to the mpb executable (necessary on mac yosemite)
# 'mpb' can be used on linux
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/documents/W1_3D_v1.ctl.txt" # 3D
# inputFile = "/Users/sean/documents/W1_2D_v03.ctl.txt" # 2D

# absolute path to the output ctl
# outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/results/BGP-2-4.txt"
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/results/BGP-1-8-3D.txt" # 3D


w1 = 0.05
w2 = 200
w3 = 0
w4 = 0.1
w5 = 0.002

weights = [ w1, w2, w3, w4, w5]
solution = {'p2': 0.15849807199408553, 'p3': 0.04275554553220376, 'p1': 0.01872000000000114, 'r1': 0.35388967134161653, 'r2': 0.41011788467137533, 'r3': 0.3028126200597566, 's3': 0, 's2': 0.044937022083806025, 's1': 0.09173700811827509}


# an experiment is just a class representation of a command line execution of mpb
# the experiment (instance) is reused between different command line calls, but
# but the command line parameters are changed between calls
# see the experiment.py module for more details
experiment = Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # currently only type 4 works for computing
experiment.setBand(23)


# parse the objParams from the experiment
# see mpbParser.py for the definition of this set of values
# (including parsing failure conditions)
objParams = parseObjFunctionParams3D(experiment)

if objParams == 0:
    print "Parsing failure"

    # the source of parsing failures is still undetermined
else:
    bandwidth = float("{0:.4f}".format(objParams[0]))
    ng0 = float("{0:.4f}".format(objParams[1]))
    avgLoss = float("{0:.4f}".format(objParams[2]))
    bgp = float("{0:.4f}".format(objParams[3]))
    loss_at_ng0 = float("{0:.4f}".format(objParams[4]))


    # in the case where bandwidth/bgp is undetermined/ too small to aproximate
    # we replace bandwidth/bgp with a tiny value, so that we do not divide by 0 in our objective function.
    if bandwidth == 0:
        bandwidth = 0.00000001
                
    if bgp == 0:
        bgp = 0.00000001


    # the weights list is defined in the execution of runExperiment.py / runOptimizer.py
    # weight for bandwidth
    w1 = weights[0]
    # weight for group index
    w2 = weights[1]
    # weight for average loss
    w3 = weights[2]
    # weight for bandwidth-group_index product
    w4 = weights[3]
    # weight for loss at ng0
    w5 = weights[4]


    # evaluate weighted sum objected function and return
    solution_score = float("{0:.4f}".format(math.sqrt((w1/bandwidth)**2 + (w2/ng0)**2 + (w3*avgLoss)**2 + ((w5*loss_at_ng0)**2) + (w4/bgp)**2)))

    print "\nSolution: " + str(solution)
    print "\nScore: " + str(solution_score)
    print "\nNormalized Bandwidth: " +  str(bandwidth)
    print "\nGroup Index: " + str(ng0)
    print "\nAverage Loss: " + str(avgLoss)
    print "\nLoss at Group Index: " + str(loss_at_ng0)
    print "\nBGP: " + str(bgp)

    
