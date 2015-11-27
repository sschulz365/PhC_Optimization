# runExperiment
# a utility for quickly testing optimized solutions in MPB

from experiment import Experiment
from objectiveFunctions import WeightedSumObjectiveFunction, WeightedSumObjectiveFunction3D, IdealDifferentialObjectiveFunction
import time

# paramMap is a dictionary that maps a string to a float value
# for each entry, the string is the name of a parameter specified in the
# mpb ctl file by the form define-param
# for each entry, the value associated to the string is what this parameter
# will be set to upon execution of mpb (as in the experiment module)

start = time.time()
print "Evaluating Solution..."
paramMap = {}
paramMap["s1"] = -0.117 # First row horizontal shift
paramMap["s2"] = 0.039 # Second row vertical shift
paramMap["s3"] = 0 # Third row vertical shift
paramMap["p1"] = 0 # First row horizontal shift
paramMap["p2"] = 0 # Second row horizontal shift
paramMap["p3"] = 0 # Third row horizontal shift
paramMap["r0"] = 0.27 # Default air-hole radius
paramMap["r1"] = 0.27 # Default first row radius
paramMap["r2"] = 0.27  # Default second row radius
paramMap["r3"] = 0.27 # Default third row radius

# absolute path to the mpb executable (necessary on mac yosemite)
# "mpb" can be used on linux
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt" # 2D
#inputFile = "/Users/sean/documents/W1_2D_5ROW.ctl.txt" # 2D

# absolute path to the output ctl
# outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/results/BGP-2-4-3D.txt"# 3D
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/results/loss_test.txt" 


# an experiment is just a class representation of a command line execution of mpb
# the experiment (instance) is reused between different command line calls, but
# but the command line parameters are changed between calls
# see the experiment.py module for more details
experiment = Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # currently only type 4 works for computing
experiment.setBand(23)

# Ideal Differntial Objective Function Weights

w1 = 0 #0.01 # bandwidth weight
w2 = 25 #100 # group index weight
w3 = 0 # average loss weight
w4 = 0 # BGP weight
w5 = 0.01 #0.002 # loss at ngo (group index) weight 
w6 = 100

weights = [ w1, w2, w3, w4, w5, w6]

# copy and paste the solution map here (in the form of paramMap)
solution = paramMap #{'p2': 0.013548, 'p3': 0.193379, 'p1': 0.014183, 'r0': 0.201901, 'r1': 0.209662, 'r2': 0.330292, 'r3': 0.242812, 's3': 0.055118, 's2': 0.074526, 's1': 0.041513}

for key in solution.keys():
    solution[key] = float("{0:.8f}".format(solution[key]))

# Here we re-evalute a solution and display the desired values such as Score, Group Index, Bandwidth, ...

# the order of results is defined in the weightedSumObjectiveFunction method
# see the objectiveFunctions.py module
# results = weightedSumObjectiveFunction(weights, solution, experiment)

ideal_group_index = 40 #self.ideal_solution[0]
ideal_bandwidth = 0.04 #self.ideal_solution[1]
ideal_loss_at_group_index = 40 #self.ideal_solution[2]
ideal_bgp = 0.7 #self.ideal_solution[3]
ideal_delay = 350


ideal = [ideal_group_index, ideal_bandwidth, ideal_loss_at_group_index, ideal_bgp, ideal_delay]
    
#results = WeightedSumObjectiveFunction3D(weights, solution, experiment) # 3D 
results = IdealDifferentialObjectiveFunction(weights, experiment, ideal).evaluate(solution) # 2D

solution_score = results[0]
bandwidth = results[1]
group_index = results[2]
avgLoss = results[3]
bandwidth_group_index_product = results[4]
loss_at_ng0 = results[5]
delay = results[6]
print "\nSolution: " + str(solution)
print "\nScore: " + str(solution_score)
print "\nNormalized Bandwidth: " +  str(bandwidth)
print "\nGroup Index: " + str(group_index)
print "\nAverage Loss: " + str(avgLoss)
print "\nLoss at Group Index: " + str(loss_at_ng0)
print "\nBGP: " + str(bandwidth_group_index_product)
print "\nDelay: " + str(delay)

end = time.time()

print "Time elapsed: " + str(end- start)