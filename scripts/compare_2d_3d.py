__author__ = 'sean'

from backend.experiment import Experiment
from backend import mpbParser

# absolute path to the mpb executable (necessary on mac yosemite)
# 'mpb' should instead be used on linux
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

solution = {'r0': 0.214376, 'r1': 0.2, 'r2': 0.2845, 'r3': 0.290696, 's3': -0.004733, 's2': -0.000407, 's1': -0.048862}

print solution
print "\n2D LOSS\n"
# absolute path to the 2D ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt"
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/loss_verified_2d.txt"

experiment = Experiment(mpb, inputFile, outputFile)
experiment.setParams(solution)
experiment.setCalculationType(4)
experiment.setBand(23)
experiment.kinterp = 19

# output file checker
#print experiment.extractFunctionParams()

#the following can be used if the outputFile has already been generated
print mpbParser.parseObjFunctionParams(experiment)

print "\n3D Loss\n"


# absolute path to the 3D ctl
inputFile = "/Users/sean/Documents/W1_3D_v1.ctl.txt"
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/loss_verified_3d_2.txt"

experiment = Experiment(mpb, inputFile, outputFile)
experiment.setParams(solution)
experiment.setCalculationType(4)
experiment.setBand(23)
experiment.kinterp = 19
experiment.dim3 = True

# output file checker
#print experiment.extractFunctionParams()

#the following can be used if the outputFile has already been generated
print mpbParser.parseObjFunctionParams3D(experiment)

