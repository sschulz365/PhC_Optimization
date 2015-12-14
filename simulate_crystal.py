__author__ = 'sean'
from experiment import Experiment
import mpbParser

# absolute path to the mpb executable (necessary on mac yosemite)
# 'mpb' should instead be used on linux
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
#inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt" # 2D
#inputFile = "/Users/sean/documents/W1_2D_5ROW.ctl.txt" # 5 hole
inputFile = "/Users/sean/Documents/W1_3D_v1.ctl.txt"



outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/Delay_verified_3d_1.txt"
#outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/results/test_2d.txt"
# absolute path to the output ctl
# outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/results/BGP-2-4.txt"



#solution = {'r0': 0.2, 'r1': 0.218358, 'r2': 0.281751, 'r3': 0.310675, 's3': -0.003429, 's2': 0.008228, 's1': -0.006247}
solution = {'r0': 0.24906, 'r1': 0.2, 'r2': 0.205319, 'r3': 0.249118, 's3': -0.004591, 's2': -0.000385, 's1': -0.035766}

#solution = {'r0': 0.286, 'r2': 0.24, 's2': 0.08, 's1': -0.10}
#solution = {'r0': 0.2, 'r1': 0.222577, 'r2': 0.267186, 'r3': 0.261162, 's3': -0.003447, 's2': 0.004093, 's1': -0.070646}

	#0.2 0.2	0.283094	0.261162	-0.006669	0.003112	-0.058935
# an experiment is just a class representation of a command line execution of mpb
# the experiment (instance) is reused between different command line calls,
# but the command line parameters are changed between calls
# see the experiment.py module for more details
experiment = Experiment(mpb, inputFile, outputFile)
experiment.setParams(solution)
experiment.setCalculationType(4)
experiment.setBand(23)
experiment.kinterp = 19
experiment.dim3 = True
#experiment.noSplit() # this command toggles whether to use mpb-split or not

print "Simulating"
# command line execute + parse
#print experiment.extractFunctionParams()
print solution
print "\n"
# output file checker
print mpbParser.parseObjFunctionParams3D(experiment)
print "\n"


# do a second computation

solution_2 = {'r0': 0.224803, 'r1': 0.2, 'r2': 0.2, 'r3': 0.261162, 's3': -0.006669, 's2': 0.003713, 's1': -0.045961}
output_2 =  "/Users/sean/UniversityOfOttawa/Photonics/PCWO/Delay_verified_3d_2.txt"

experiment2 = Experiment(mpb, inputFile, output_2)
experiment2.setParams(solution_2)
experiment2.setCalculationType(4)
experiment2.setBand(23)
experiment2.kinterp = 19

print solution_2
print "\n"

# command line execute + parse
print experiment2.extractFunctionParams()
print solution
# output file checker
#print mpbParser.parseObjFunctionParams3D(experiment2)
print "\n"
