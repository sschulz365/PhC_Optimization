#generate Feild Profiles


from backend.experiment import Experiment
import subprocess
import matplotlib.pyplot as plt
import csv
import math
import subprocess

print "Generating Solution Field Profiles..."

# absolute path to the mpb executable (necessary on mac yosemite)
# 'mpb' can be used on linux
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/W1_2D_v04.ctl.txt" # 2D
#inputFile = "/Users/sean/documents/W1_2D_5ROW.ctl.txt" # 5 hole

outputFilePrefix = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/delay_1500"
# absolute path to the output ctl
# outputFile = "/Users/sean/UniversityOfOttawa/Photonics/MPBproject/results/BGP-2-4.txt"
outputField = outputFilePrefix + "_Field.txt" # 3D



solution = {'r0': 0.273167, 'r1': 0.2, 'r2': 0.2, 'r3': 0.249118, 's3': -0.003447, 's2': 0.004997, 's1': 0.040613}


# an experiment is just a class representation of a command line execution of mpb
# the experiment (instance) is reused between different command line calls,
# but the command line parameters are changed between calls
# see the experiment.py module for more details
experiment = Experiment(mpb, inputFile, outputField)
experiment.setParams(solution)
experiment.setCalculationType(3) 
experiment.setBand(23)
#experiment.noSplit() # this command toggles whether to use mpb-split or not
experiment.perform()



outputCSV = outputFilePrefix + ".te.csv"

print "\nGenerating output field at ", outputCSV

experiment.setCalculationType(4) # currently only type 4 works for computing
experiment.outputFile = outputFilePrefix + ".txt"  
experiment.perform()

subprocess.call("grep tefreqs " + experiment.outputFile + " > " + outputCSV,shell=True)



print "\nGenerating Group Index Plot..."

inputText = outputFilePrefix + ".txt"
outputGV = outputFilePrefix + ".gv.csv"



subprocess.call("grep tevelocity " + inputText + " > " + outputGV,shell=True)



band = 23
velocities = []
groupIndexMap = {}
count = {}
with open(outputGV, 'rb') as csvfile:
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
    
    ng0 = groupIndexMap[minIndex] # central group index
    #print "Group Index " + str(ng0) + " at " + str(minIndex)
    #print "avg group index? " + str(sum(groupIndexMap.values())/len(groupIndexMap.values()))
    
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
