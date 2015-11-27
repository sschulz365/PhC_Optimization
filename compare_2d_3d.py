__author__ = 'sean'
from experiment import Experiment
import mpbParser

# absolute path to the mpb executable (necessary on mac yosemite)
# 'mpb' should instead be used on linux
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

solution = {'r0': 0.214376, 'r1': 0.2, 'r2': 0.2845, 'r3': 0.290696, 's3': -0.004733, 's2': -0.000407, 's1': -0.048862}

print solution
print "\n2D LOSS\n"
# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt"
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/loss_verified_2d.txt"

experiment = Experiment(mpb, inputFile, outputFile)
experiment.setParams(solution)
experiment.setCalculationType(4)
experiment.setBand(23)
experiment.kinterp = 19

# output file checker
#print experiment.extractFunctionParams()
print mpbParser.parseObjFunctionParams(experiment)

print "\n3D Loss\n"


#inputFile = "/Users/sean/documents/W1_2D_5ROW.ctl.txt" # 5 hole
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
print mpbParser.parseObjFunctionParams3D(experiment)

print "\n"

solution_2 = {'p2': 0.005009, 'p3': 0.006596, 'p1': -0.115785, 'r0': 0.237322, 'r1': 0.2, 'r2': 0.383745, 'r3': 0.2, 's3': -0.046694, 's2': -0.101481, 's1': -0.062471}
print solution_2
print "\n2D GBP\n"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt"
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/GBP_verified_2d_2.txt"

experiment = Experiment(mpb, inputFile, outputFile)
experiment.setParams(solution_2)
experiment.setCalculationType(4)
experiment.setBand(23)
experiment.kinterp = 19

# output file checker
print experiment.extractFunctionParams()
#print mpbParser.parseObjFunctionParams(experiment)
print "\n3D GBP\n"


#inputFile = "/Users/sean/documents/W1_2D_5ROW.ctl.txt" # 5 hole
inputFile = "/Users/sean/Documents/W1_3D_v1.ctl.txt"
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/GBP_verified_3d_2.txt"

experiment = Experiment(mpb, inputFile, outputFile)
experiment.setParams(solution_2)
experiment.setCalculationType(4)
experiment.setBand(23)
experiment.kinterp = 19
experiment.dim3 = True

# output file checker
#print experiment.extractFunctionParams()
print mpbParser.parseObjFunctionParams3D(experiment)

print "\n"
