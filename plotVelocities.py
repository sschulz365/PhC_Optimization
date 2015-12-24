# Requirements


import matplotlib.pyplot as plt
import csv
import math
import subprocess

inputfile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/GBP_verified_3d_8.txt"
outputfile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/GBP_verified_3d_8.gv.csv"
if "2d" in inputfile:
    subprocess.call("grep tevelocity " + inputfile + " > " + outputfile,shell=True)
else:
    subprocess.call("grep zevenvelocity " + inputfile + " > " + outputfile,shell=True)

band = 23
velocities = []
groupIndexMap = {}
count = {}
with open(outputfile, 'rb') as csvfile:
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
