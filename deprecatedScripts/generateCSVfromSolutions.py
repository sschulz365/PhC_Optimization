
from backend.experiment import Experiment
from backend import mpbParser


#Delay Solutions
population = [ {'p2': 0.473642, 'p3': 0.035064, 'p1': 0.082575, 'r0': 0.302144, 'r1': 0.212816, 'r2': 0.222178, 'r3': 0.205978, 's3': 0.032043, 's2': 0.080835, 's1': 0.128952},
    {'p2': 0.021263, 'p3': 0.02073, 'p1': 0.068545, 'r0': 0.276923, 'r1': 0.271703, 'r2': 0.334839, 'r3': 0.264194, 's3': 0.018017, 's2': 0.035167, 's1': 0.114101},
    {'p2': 0.024356, 'p3': 0.016564, 'p1': 0.05915, 'r0': 0.2103, 'r1': 0.243295, 'r2': 0.326217, 'r3': 0.257731, 's3': 0.01748, 's2': 0.026789, 's1': 0.089518},
    {'p2': 0.024356, 'p3': 0.019506, 'p1': 0.062017, 'r0': 0.2, 'r1': 0.2, 'r2': 0.336475, 'r3': 0.215609, 's3': 0.015539, 's2': 0.038054, 's1': 0.098137},
    {'p2': 0.045707, 'p3': 0.112149, 'p1': 0.00669, 'r0': 0.228166, 'r1': 0.263885, 'r2': 0.219751, 'r3': 0.245569, 's3': 0.129748, 's2': 0.374036, 's1': 0.008352},
    {'p2': 0.061008, 'p3': 0.015413, 'p1': 0.074686, 'r0': 0.2, 'r1': 0.219068, 'r2': 0.350905, 'r3': 0.297965, 's3': 0.165047, 's2': 0.06357, 's1': 0.024844},
    {'p2': 0.09035, 'p3': 0.032399, 'p1': 0.008474, 'r0': 0.2, 'r1': 0.237095, 'r2': 0.285693, 'r3': 0.293842, 's3': 0.029005, 's2': 0.002644, 's1': 0.036666},
    {'p2': 0.026477, 'p3': 0.017797, 'p1': 0.079215, 'r0': 0.276923, 'r1': 0.368515, 'r2': 0.331251, 'r3': 0.264194, 's3': 0.013987, 's2': 0.032806, 's1': 0.12658},
    {'p2': 0.09035, 'p3': 0.040395, 'p1': 0.008987, 'r0': 0.236982, 'r1': 0.2, 'r2': 0.293876, 'r3': 0.293842, 's3': 0.033493, 's2': 0.00418, 's1': 0.041612},
    {'p2': 0.087886, 'p3': 0.032399, 'p1': 0.008474, 'r0': 0.2, 'r1': 0.23588, 'r2': 0.298452, 'r3': 0.223631, 's3': 0.033493, 's2': 0.002644, 's1': 0.035414},
    {'p2': 0.036733, 'p3': 0.08568, 'p1': 0.00767, 'r0': 0.217225, 'r1': 0.325867, 'r2': 0.238558, 'r3': 0.2, 's3': 0.16589, 's2': 0.374036, 's1': 0.088698},
    {'p2': 0.034398, 'p3': 0.090089, 'p1': 0.006363, 'r0': 0.221698, 'r1': 0.266252, 'r2': 0.224129, 'r3': 0.245141, 's3': 0.176958, 's2': 0.374036, 's1': 0.088698},
    {'p2': 0.082356, 'p3': 0.033078, 'p1': 0.008347, 'r0': 0.2, 'r1': 0.252112, 'r2': 0.2, 'r3': 0.4, 's3': 0.031643, 's2': 0.002806, 's1': 0.043316},
    {'p2': 0.113727, 'p3': 0.033078, 'p1': 0.007546, 'r0': 0.201461, 'r1': 0.211033, 'r2': 0.2, 'r3': 0.332535, 's3': 0.038549, 's2': 0.003899, 's1': 0.032729},
    {'p2': 0.113727, 'p3': 0.033078, 'p1': 0.007546, 'r0': 0.226401, 'r1': 0.211033, 'r2': 0.2, 'r3': 0.332535, 's3': 0.033297, 's2': 0.00484, 's1': 0.043316},
    {'p2': 0.113727, 'p3': 0.026601, 'p1': 0.007786, 'r0': 0.201461, 'r1': 0.211033, 'r2': 0.2, 'r3': 0.332535, 's3': 0.042909, 's2': 0.003395, 's1': 0.043316},
    {'p2': 0.073352, 'p3': 0.049478, 'p1': 0.007446, 'r0': 0.238271, 'r1': 0.252112, 'r2': 0.235449, 'r3': 0.38496, 's3': 0.032163, 's2': 0.003769, 's1': 0.039224},
    {'p2': 0.113727, 'p3': 0.033078, 'p1': 0.007546, 'r0': 0.201461, 'r1': 0.211033, 'r2': 0.2, 'r3': 0.332535, 's3': 0.04009, 's2': 0.004055, 's1': 0.043316},
    {'p2': 0.113727, 'p3': 0.02643, 'p1': 0.010409, 'r0': 0.2, 'r1': 0.208881, 'r2': 0.237628, 'r3': 0.320288, 's3': 0.043522, 's2': 0.004545, 's1': 0.039821},
    {'p2': 0.113727, 'p3': 0.033078, 'p1': 0.007546, 'r0': 0.226585, 'r1': 0.211033, 'r2': 0.22736, 'r3': 0.332535, 's3': 0.04009, 's2': 0.003809, 's1': 0.043316},
    {'p2': 0.111011, 'p3': 0.033078, 'p1': 0.007523, 'r0': 0.2, 'r1': 0.210612, 'r2': 0.2, 'r3': 0.332535, 's3': 0.04009, 's2': 0.003395, 's1': 0.043316},
    {'p2': 0.113727, 'p3': 0.033078, 'p1': 0.009762, 'r0': 0.2, 'r1': 0.211033, 'r2': 0.2, 'r3': 0.332535, 's3': 0.029901, 's2': 0.003395, 's1': 0.043316},
    {'p2': 0.118272, 'p3': 0.033078, 'p1': 0.006623, 'r0': 0.2, 'r1': 0.237869, 'r2': 0.200129, 'r3': 0.373222, 's3': 0.040061, 's2': 0.002346, 's1': 0.046705},
    {'p2': 0.090721, 'p3': 0.031196, 'p1': 0.007345, 'r0': 0.2, 'r1': 0.241899, 'r2': 0.237691, 'r3': 0.373222, 's3': 0.040061, 's2': 0.002346, 's1': 0.037974},
    {'p2': 0.113727, 'p3': 0.034615, 'p1': 0.009597, 'r0': 0.2, 'r1': 0.27989, 'r2': 0.242626, 'r3': 0.4, 's3': 0.04355, 's2': 0.00217, 's1': 0.036666},
    {'p2': 0.083944, 'p3': 0.051006, 'p1': 0.006514, 'r0': 0.2, 'r1': 0.211731, 'r2': 0.260247, 'r3': 0.333946, 's3': 0.040068, 's2': 0.003086, 's1': 0.036666},
    {'p2': 0.073352, 'p3': 0.04082, 'p1': 0.006514, 'r0': 0.201461, 'r1': 0.252112, 'r2': 0.260247, 'r3': 0.38496, 's3': 0.032163, 's2': 0.003769, 's1': 0.036666},
    {'p2': 0.432663, 'p3': 0.036676, 'p1': 0.082575, 'r0': 0.302144, 'r1': 0.212816, 'r2': 0.2, 'r3': 0.205978, 's3': 0.029867, 's2': 0.080835, 's1': 0.128952},
    {'p2': 0.107285, 'p3': 0.179389, 'p1': 0.00767, 'r0': 0.249148, 'r1': 0.325867, 'r2': 0.4, 'r3': 0.241423, 's3': 0.16589, 's2': 0.05722, 's1': 0.088698},
    {'p2': 0.478775, 'p3': 0.04547, 'p1': 0.054618, 'r0': 0.307939, 'r1': 0.212816, 'r2': 0.2, 'r3': 0.241448, 's3': 0.025433, 's2': 0.113949, 's1': 0.128952},
    ]



# absolute path to the mpb executable
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt"

# absolute path to the output ctl
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/csvTestFile.txt"

csv_file = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/delay_solutions.txt"

# we define a generalized experiment object
# that we reuse whenever we need to make a command-line mpb call
# see experiment.py for functionality
experiment = Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # accepts an int from 0 to 5
experiment.setBand(23)


out_stream = open(csv_file, 'w')
out_stream.write("Delay, BGP, Group Index, Bandwidth, r0, r1, r2, r3, s1, s2, s3, p1, p2, p3")
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
    bandwidth_group_index_product = results[3] #BGP
    loss_at_ng0 = results[4] # loss at group index
    delay = results[5]

    print "\nNormalized Bandwidth: " +  str(bandwidth)
    print "\nGroup Index: " + str(-group_index)
    print "\nAverage Loss: " + str(avgLoss)
    print "\nLoss at Group Index: " + str(loss_at_ng0)
    print "\nBGP: " + str(bandwidth_group_index_product)
    print "\nDelay: " + str(delay) + "\n"

    out_string = str(delay) + ", " + str(bandwidth_group_index_product) + ", " + str(group_index) + ", " + str(bandwidth) + ", "
    out_string = out_string + str(solution["r0"]) + ", " + str(solution["r1"]) + ", " + str(solution["r2"]) +  ", " + str(solution["r3"]) + ", "
    out_string = out_string + str(solution["s1"]) + ", " + str(solution["s2"]) + ", " + str(solution["s3"]) + ", "
    out_string = out_string + str(solution["p1"]) + ", " + str(solution["p2"]) + ", " + str(solution["p3"]) + "\n"

    out_stream.write(out_string)

print "CSV generated"

out_stream.close()