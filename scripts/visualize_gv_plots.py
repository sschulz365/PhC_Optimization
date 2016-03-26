__author__ = 'sean'



# Requirements


import matplotlib.pyplot as plt
import csv
import math
import subprocess

from backend.experiment import Experiment
import backend.mpbParser

# absolute path to the mpb executable (necessary on mac yosemite)
# 'mpb' should instead be used on linux
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/W1_2D_v04.ctl.txt" # 2D

# absolute path to the output .txt
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/visualize_GBP_2d.txt"
# output cv destination

outputcv = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/visualize_GBP_2d.gv.csv"

#solution = {'r0': 0.2, 'r1': 0.218358, 'r2': 0.281751, 'r3': 0.310675, 's3': -0.003429, 's2': 0.008228, 's1': -0.006247}
solutions = [{'p2': 0.305676, 'p3': -0.001217, 'p1': 0.226142, 'r0': 0.20691600000000002, 'r1': 0.20001000000000002, 'r2': 0.277464, 'r3': 0.227303, 's3': 0.02538, 's2': 0.161438, 's1': 0.198196},
             {'p2': -0.042182, 'p3': 0.005852, 'p1': -0.135591, 'r0': 0.233088, 'r1': 0.200446, 'r2': 0.383809, 'r3': 0.200118, 's3': -0.065357, 's2': -0.07563, 's1': -0.069276},
             {'p2': 0.178789, 'p3': 0.030898, 'p1': 0.124789, 'r0': 0.22809, 'r1': 0.237711, 'r2': 0.4, 'r3': 0.200028, 's3': 0.076132, 's2': -0.097793, 's1': 0.063853},
             {'p2': -0.12578699999999998, 'p3': -0.050950999999999996, 'p1': 0.13126800000000002, 'r0': 0.20001000000000002, 'r1': 0.20001000000000002, 'r2': 0.280337, 'r3': 0.259042, 's3': -0.052989999999999995, 's2': -0.141012, 's1': -0.15109299999999998},
{'p2': 0.005054, 'p3': 0.004457, 'p1': -0.135639, 'r0': 0.232882, 'r1': 0.20002, 'r2': 0.383711, 'r3': 0.2, 's3': -0.065592, 's2': -0.075703, 's1': -0.069698},
{'p2': 0.211282, 'p3': -0.042654, 'p1': 0.124609, 'r0': 0.20001, 'r1': 0.2, 'r2': 0.310886, 'r3': 0.2, 's3': -0.078041, 's2': -0.095444, 's1': -0.13245},
{'p2': 0.02888, 'p3': 0.028308, 'p1': -0.141661, 'r0': 0.237307, 'r1': 0.202992, 'r2': 0.38373, 'r3': 0.2, 's3': -0.046709, 's2': -0.10155, 's1': -0.062486},
{'p2': 0.005019, 'p3': 0.006606, 'p1': -0.135408, 'r0': 0.23733200000000002, 'r1': 0.20001000000000002, 'r2': 0.383755, 'r3': 0.20001000000000002, 's3': -0.046683999999999996, 's2': -0.094848, 's1': -0.062460999999999996},
{'p2': 0.16922700000000002, 'p3': 0.030898, 'p1': 0.124609, 'r0': 0.22791, 'r1': 0.23769300000000002, 'r2': 0.40001000000000003, 'r3': 0.20001000000000002, 's3': 0.094081, 's2': -0.097833, 's1': 0.063673},
{'p2': 0.005025, 'p3': 0.006642, 'p1': -0.113161, 'r0': 0.237332, 'r1': 0.20001, 'r2': 0.383809, 'r3': 0.200064, 's3': -0.046648, 's2': -0.098852, 's1': -0.062419},
{'p2': -0.183304, 'p3': -0.050531999999999994, 'p1': -0.14166099999999998, 'r0': 0.20001000000000002, 'r1': 0.260614, 'r2': 0.355705, 'r3': 0.20001000000000002, 's3': 0.08078099999999999, 's2': -0.09217800000000001, 's1': -0.13489299999999999},
{'p2': -0.209385, 'p3': 8.499999999999999e-05, 'p1': -0.152728, 'r0': 0.20001000000000002, 'r1': 0.20305700000000002, 'r2': 0.276104, 'r3': 0.21324300000000002, 's3': -0.020113, 's2': -0.109978, 's1': -0.10892500000000001},
{'p2': -0.21046499999999999, 'p3': -0.042651999999999995, 'p1': -0.162243, 'r0': 0.20001000000000002, 'r1': 0.20001000000000002, 'r2': 0.264392, 'r3': 0.231603, 's3': -0.070359, 's2': -0.153367, 's1': -0.1217}]
#solution = {'r0': 0.286, 'r2': 0.24, 's2': 0.08, 's1': -0.10}
#solution = {'r0': 0.2, 'r1': 0.222577, 'r2': 0.267186, 'r3': 0.261162, 's3': -0.003447, 's2': 0.004093, 's1': -0.070646}

solutions = [{'p2': 0.16922700000000002, 'p3': 0.030898, 'p1': 0.124609, 'r0': 0.22791, 'r1': 0.23769300000000002, 'r2': 0.40001000000000003, 'r3': 0.20001000000000002, 's3': 0.094081, 's2': -0.097833, 's1': 0.063673},
             {'p2': -0.209385, 'p3': 8.499999999999999e-05, 'p1': -0.152728, 'r0': 0.20001000000000002, 'r1': 0.20305700000000002, 'r2': 0.276104, 'r3': 0.21324300000000002, 's3': -0.020113, 's2': -0.109978, 's1': -0.10892500000000001}
             ]
for solution in solutions:
        #0.2 0.2	0.283094	0.261162	-0.006669	0.003112	-0.058935
    # an experiment is just a class representation of a command line execution of mpb
    # the experiment (instance) is reused between different command line calls,
    # but the command line parameters are changed between calls
    # see the experiment.py module for more details
    experiment = Experiment(mpb, inputFile, outputFile)
    experiment.setParams(solution)
    experiment.setCalculationType(4)
    experiment.setBand(23)
    experiment.kinterp = 39
    #experiment.dim3 = True
    #experiment.noSplit() # this command toggles whether to use mpb-split or not


    print "\n"
    print "Simulating"
    print solution

    # command line execute + parse
    print experiment.extractFunctionParams()




    if "3d" in outputFile:
        subprocess.call("grep zevenvelocity " + outputFile + " > " + outputcv,shell=True)
    else:
        subprocess.call("grep tevelocity " + outputFile + " > " + outputcv,shell=True)

    band = 23
    velocities = []
    groupIndexMap = {}
    count = {}
    with open(outputcv, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
                if "k index" not in row[1]:
                    #print row
                    velocity_vector = row[1 + band].replace("#(",'').replace(')','').split(' ')
                    #print velocity_vector
                    x_velocity = float(velocity_vector[1])
                    groupIndexMap[int(row[1])] = math.fabs(float(1)/x_velocity)
                    count[int(row[1])] = int(row[1])
                    y_velocity = float(velocity_vector[2][:7])
                    velocities.append(x_velocity)

        kPointCount = max(count.values())
        # print groupIndexMap
        minIndex = 0 # maybe should use a better term than index here
        minDelta = 100000000000
        for i in range(2,kPointCount): # experiment.kinterp is the number of kpoints
        #  if a given k point does not exist in the map
        #  then there was a parsing failure
        #  "Parsing Failure" should be printed by the calling method objectiveFunctions.py


            currentIndex = groupIndexMap[i]

            # right now this  > 10 criteria is hard coded, but it should be parametized at some point
            if math.fabs(currentIndex) > 10:

                prevIndex = groupIndexMap[i-1]

                nextIndex = groupIndexMap[i+1]

                # compute approximation to group index derivative
                delta1 = math.fabs(currentIndex - prevIndex)
                delta2 = math.fabs(nextIndex - currentIndex)

                thisDelta = (delta1 + delta2 ) / 2
                # print i # sanity check
                # print thisDelta

                # check for minimum delta corrseponding to the 'flat' range of the group index
                if thisDelta < minDelta:
                    minDelta = thisDelta
                    minIndex = i

        ng0 = groupIndexMap[minIndex] # central group index (but not really)

        print "avg group index? " + str(sum(groupIndexMap.values())/len(groupIndexMap.values()))

     # compute 'viable' bandwidth range based on feasible group index
        # store the bandwidth identifiers (k points) in potentialBandwidthIndexes
        # this array will contain  w where ng_i > 0.9*ng0, ng_i < 1.1*ng0
        # bandwidth is essentially an approximation to dw/w0
    potentialBandwidthIndexes = []

    for i in range(1,kPointCount + 1):
            if ((groupIndexMap[i] < 1.1*ng0) and (groupIndexMap[i] > 0.9*ng0)) or ((groupIndexMap[i] > 1.1*ng0) and (groupIndexMap[i] < 0.9*ng0)):
                potentialBandwidthIndexes.append(i)

    #sprint velocities
    plt.plot(groupIndexMap.values())
    plt.ylabel('Group Index')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,200))
    plt.show()
