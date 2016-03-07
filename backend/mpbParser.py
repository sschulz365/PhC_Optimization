# Sean Billings, 2015
import math


# The following parser has been defined for the line defect waveguide
# as specified in output format used for the W1_2D_v04.ctl mpb control file
# this method parses, Loss, bandwidth, and Group Index and associated output values like loss at ng0 and BGP
def parseObjFunctionParams(experiment): # input is an experiment
    outF = open(experiment.outputFile)
    groupIndex = 0
    loss = 0
    band = experiment.band


     # maps k points to the corresponding (te)frequency (for the band as specified in runExperiment.py, runOptimzer,py)
    bandwidths = {}

    # maps k points to the corresponding (te)velocity (for the band as specified in runExperiment.py, runOptimzer,py)
    groupVelocityMap = {}

    # maps k points to (1/ groupVelocityMap[k])
    groupIndexMap = {}


    gammaValues = {} # used to temporarily store gamma values for Loss computation (as in Dr. Schulz paper)
    rhoValues = {} # used to temporarily store rho values for Loss computation (as in Dr. Schulz paper)
    lossMap = {} # stores the loss at each k point
    lossContrastMap = {}
    contrast_gamma = 0 # used for ensuring valid loss computation
    contrast_rho = 0 # used for ensuring valid loss computation

    kPoint = -1 # defines kPoint outside of the for loop

    #iterates over the file in reverse order (the order affects the feature extraction entirely)
    for line in reversed(outF.readlines()):

        #1 search for bandwidths and assign at index accoring to the band
        if "tefreqs:, " in line and "k index" not in line:
            bandInfo = line.split(",")
            kPoint = int(bandInfo[1])
            bandwidths[kPoint] = float(bandInfo[band + 5]) # should be consistent acrosss iterations

        # 2 check for group velocity
        if "tevelocity:, " in line:
            groupVelocityLine = line.split(",")
            kPoint = int(groupVelocityLine[1].replace(" ",""))
            # print kPoint # sanity check
            groupVector = groupVelocityLine[band + 1].replace(" #(","").replace(")", "").split(" ")
            if 'e' in groupVector[0]:
                # print groupVelocityLine # sanity check
                # print groupVector[0] # sanity check
                decimal = groupVector[0].split('e-')[0]
                magnitude = float(groupVector[0].split('e-')[1])
                groupVelocityMap[kPoint] = float(decimal) / float(math.pow(10,magnitude))
            else:
                groupVelocityMap[kPoint] = groupVector[0] # x value
                
            groupIndexMap[kPoint] = (float(1)/float(groupVelocityMap[kPoint]))

        # 3 retrieve gamma and rho coefficients to compute loss
        # gamma coefficients found in the section ranging from
        # "calling integral-rho-holes-list-first-row" to "solve_kpoint"
        
        if "integral-gamma-holes-list-third-row" in line and not "Calling" in line:
            # print "[DEBUG}" + str(line)  # sanity check
            # print "[DEBUG]" + str(line.split(",")[20].replace("+0.0i", "")) # sanity check
            # There was a bug here regarding invalid float value: 0, but it seams to have disappeared
            # it potentially may have been caused by threading errors in mpb-split corrupting the output
            if (line.split(",")[20].replace("+0.0i", "")) != "0":
                gammaValues[3] = float(line.split(",")[20].replace("+0.0i", ""))
            else:
                gammaValues[3] = 0
        if "integral-gamma-holes-list-second-row" in line and not "Calling" in line:
            if (line.split(",")[20].replace("+0.0i", "")) != "0":
                gammaValues[2] = float(line.split(",")[20].replace("+0.0i", ""))
            else:
                gammaValues[2] = 0
        if "integral-gamma-holes-list-first-row" in line and not "Calling" in line:
            if (line.split(",")[20].replace("+0.0i", "")) != "0":
                gammaValues[1] = float(line.split(",")[20].replace("+0.0i", ""))
            else:
                gammaValues[1] = 0

        # measures the electric field extended into the crystal to detect mpb loss anomalies
        if "integral-gamma-holes-list-sixth-row" in line and not "Calling" in line:
            if (line.split(",")[20].replace("+0.0i", "")) != "0":
                contrast_gamma = float(line.split(",")[20].replace("+0.0i", ""))
            else:
                contrast_gamma = 0

        if "integral-rho-holes-list-third-row" in line and not "Calling" in line:
            rhoValues[3] = float(line.split(",")[16])
        if "integral-rho-holes-list-second-row" in line and not "Calling" in line:
            rhoValues[2] = float(line.split(",")[16])
        if "integral-rho-holes-list-first-row" in line and not "Calling" in line:
            rhoValues[1] = float(line.split(",")[16])

        # measures the electric field extended into the crystal to detect mpb loss anomalies
        if "integral-rho-holes-list-sixth-row" in line and not "Calling" in line:
            contrast_rho = float(line.split(",")[16])

        # this line will occur after the above,
        # at this point loss can be computed (as in Dr. Schulz Paper)
        if ("solve_kpoint" in line):
            if (not gammaValues == {}) and (not rhoValues == {}) and band != -1:
                gamma = sum(gammaValues.values())*experiment.fieldFraction
                rho = sum(rhoValues.values())*experiment.fieldFraction
                contrast_gamma = contrast_gamma*experiment.fieldFraction
                contrast_rho = contrast_rho*experiment.fieldFraction
                groupIndex = groupIndexMap[kPoint] 
                lossMap[kPoint] = experiment.c1*math.fabs(groupIndex)*gamma + experiment.c2*(groupIndex**2)*rho
                lossContrastMap[kPoint] = experiment.c1*math.fabs(groupIndex)*contrast_gamma + experiment.c2*(groupIndex**2)*contrast_rho
                # reset gammvalues and rhoValues
                gammaValues = {}
                rhoValues = {}
                contrast_gamma = 0
                contrast_rho = 0
                #the loss formula is from "Beyond the effective index method: ..." by Dr. Schulz et al

    #end for (actual parsing)

    print "DEBUG: Loss Map: " + str(lossMap)
    print "DEBUG: Group Index Map: " + str(groupIndexMap)

    return extractFOM(experiment,lossMap, lossContrastMap, groupIndexMap,bandwidths,"MAXGBP")



# The following parser has been defined for the line defect waveguide
# as specified in output format used for the W1_3D_v1.ctl mpb control file
# this method parses, Loss, bandwidth, and Group Index and associated output values like loss at ng0 and BGP
def parseObjFunctionParams3D(experiment): # input is an experiment
    outF = open(experiment.outputFile)
    groupIndex = 0
    loss = 0
    band = experiment.band


     # maps k points to the corresponding (te)frequency (for the band as specified in runExperiment.py, runOptimzer,py)
    bandwidths = {}

    # maps k points to the corresponding (te)velocity (for the band as specified in runExperiment.py, runOptimzer,py)
    groupVelocityMap = {}

    # maps k points to (1/ groupVelocityMap[k])
    groupIndexMap = {}


    gammaValues = {} # used to temporarily store gamma values for Loss computation (as in Dr. Schulz paper)
    rhoValues = {} # used to temporarily store rho values for Loss computation (as in Dr. Schulz paper)
    lossMap = {} # stores the loss at each k point
    lossContrastMap = {}
    kPoint = -1 # defines kPoint outside of the for loop

    #iterates over the file in reverse order (the order affects the feature extraction entirely)
    for line in reversed(outF.readlines()):
        # regex may need to be fixed

        #1 search for bandwidths and assign at index accoring to the band
        if "zevenfreqs:, " in line and "k index" not in line:
            bandInfo = line.split(",")
            kPoint = int(bandInfo[1])
            bandwidths[kPoint] = float(bandInfo[band + 5]) # should be consistent acrosss iterations
        
        # 2 check for group velocity
        if "zevenvelocity:, " in line:
            groupVelocityLine = line.split(",")
            kPoint = int(groupVelocityLine[1].replace(" ",""))
            # print kPoint # sanity check
            groupVector = groupVelocityLine[band + 1].replace(" #(","").replace(")", "").split(" ")
            if 'e' in groupVector[0]:
                # print groupVelocityLine # sanity check
                # print groupVector[0] # sanity check
                decimal = groupVector[0].split('e-')[0]
                magnitude = float(groupVector[0].split('e-')[1])
                groupVelocityMap[kPoint] = float(decimal) / float(math.pow(10,magnitude))
            else:
                groupVelocityMap[kPoint] = groupVector[0] # x value
                
            groupIndexMap[kPoint] = (float(1)/float(groupVelocityMap[kPoint]))

        # 3 retrieve gamma and rho coefficients to compute loss
        # gamma coefficients found in the section ranging from
        # "calling integral-rho-holes-list-first-row" to "solve_kpoint"
        
        if "integral-gamma-holes-list-third-row" in line and not "Calling" in line:
            # print line  # sanity check
            gammaValues[3] = float(line.split(",")[len(line.split(","))-1].replace("+0.0i", ""))
        if "integral-gamma-holes-list-second-row" in line and not "Calling" in line:
            gammaValues[2] = float(line.split(",")[len(line.split(","))-1].replace("+0.0i", ""))
        if "integral-gamma-holes-list-first-row" in line and not "Calling" in line:
            gammaValues[1] = float(line.split(",")[len(line.split(","))-1].replace("+0.0i", ""))

        if "integral-rho-holes-list-third-row" in line and not "Calling" in line:
            rhoValues[3] = float(line.split(",")[len(line.split(","))-1])
        if "integral-rho-holes-list-second-row" in line and not "Calling" in line:
            rhoValues[2] = float(line.split(",")[len(line.split(","))-1])
        if "integral-rho-holes-list-first-row" in line and not "Calling" in line:
            rhoValues[1] = float(line.split(",")[len(line.split(","))-1])

        # this line will occur after the above,
        # at this point loss can be computed (as in Dr. Schulz Paper)
        if ("solve_kpoint" in line):
            # print line

            if (not gammaValues == {}) and (not rhoValues == {}) and band != -1:
                # no field fraction for 3d simulations
                gamma = sum(gammaValues.values())
                rho = sum(rhoValues.values())
                groupIndex = groupIndexMap[kPoint] 
                lossMap[kPoint] = experiment.c1*groupIndex*gamma + experiment.c2*(groupIndex**2)*rho
                lossContrastMap[kPoint] = 0
                # reset gammvalues and rhoValues
                gammaValues = {}
                rhoValues = {}
                #the loss formula is from "Beyond the effective index method: ..." by Dr. Schulz et al
            
        
            

        
    #end for (actual parsing)

    print "Loss Map: " + str(lossMap)
    print "Group Index Map: " + str(groupIndexMap)

    return extractFOM(experiment,lossMap, lossContrastMap, groupIndexMap, bandwidths, "MAXGBP")

def extractFOM(experiment, lossMap, lossContrastMap, groupIndexMap, bandwidths, mode):
     # The following section determines the group index
    maxBandwidthRatio = 0
    ng0_index = 0
    # loss_constraint = 0.7
    print "\n"
    for j in range(2,experiment.kinterp): # experiment.kinterp is the number of kpoints
        #  if a given k point does not exist in the map
        #  then there was a parsing failure
        #  "Parsing Failure" should be printed by the calling method objectiveFunctions.py
        if j not in groupIndexMap.keys():
            print groupIndexMap.keys()
            print "Parsing failure"
            output_map = {}
            output_map["bandwidth"] = 0
            output_map["ng0"] = 0
            output_map["avgLoss"] = 100000
            output_map["GBP"] = 0
            output_map["loss_at_ng0"]= 100000
            output_map["delay"] = 0
            return output_map


        test_ng0 = groupIndexMap[j]



        if lossContrastMap[j] < 0.1*lossMap[j]:

            potentialBandwidthIndexes = []

            if math.fabs(test_ng0) > 10:

                for i in range(1,experiment.kinterp + 1):
                    if (((groupIndexMap[i] < 1.1*test_ng0) and (groupIndexMap[i] > 0.9*test_ng0)) or ((groupIndexMap[i] > 1.1*test_ng0) and (groupIndexMap[i] < 0.9*test_ng0))) and lossContrastMap[i] < 0.1*lossMap[i]:
                        potentialBandwidthIndexes.append(i)

                # print potentialBandwidthIndexes # sanity check
                # trim potentialBandwidthIndexes to a 'continuous' set of values
                if len(potentialBandwidthIndexes) == 0:
                    nextBandWidthRatio = 0
                else:
                    bandwidthMinIndex = min(potentialBandwidthIndexes)
                    bandwidthMaxIndex = max(potentialBandwidthIndexes)
                    for m in range(bandwidthMinIndex, j): # note minIndex is the index with lowest change in ngo
                        if m not in potentialBandwidthIndexes:
                            for n in range(bandwidthMinIndex,m):
                                if n in potentialBandwidthIndexes:
                                    potentialBandwidthIndexes.remove(n)

                    startRemoving = False
                    for k in range(j, bandwidthMaxIndex + 1): # note minIndex is the index with lowest change in ngo
                        if startRemoving:
                            if k in potentialBandwidthIndexes:
                                    potentialBandwidthIndexes.remove(k)
                        else:
                            if k not in potentialBandwidthIndexes:
                                startRemoving = True
                    # print potentialBandwidthIndexes # sanity check
                    # prepare to compute average loss and bandwidth

                    # stores the (te)frequency at viable k points
                    viableBandwidths = {}

                    # stores the group index values at viable k points
                    viableGroupIndexes = {}


                    for p in potentialBandwidthIndexes:
                        viableBandwidths[p] = bandwidths[p]
                        viableGroupIndexes[p] = math.fabs(groupIndexMap[p])

                    if len(viableBandwidths) != 0:
                        # print viableBandwidths # sanity check

                        # bandwidthNormalized is essentially dw/w0 for the range of ng such that [ ng_i > 0.9*ng0 AND ng_i < 1.1*ng0]
                        bandwidthNormalized = ( max(viableBandwidths.values())
                                            - min(viableBandwidths.values()) ) / viableBandwidths[j]

                        # compute the average group index (this may not be a useful measure)
                        avgGroupIndex = sum(viableGroupIndexes.values()) / float(len(viableGroupIndexes.values()))


                        # compute BGP
                        #nextBandWidthRatio = test_ng0*bandwidthNormalized
                        # or
                        nextBandWidthRatio = avgGroupIndex*bandwidthNormalized
                    else:
                        nextBandWidthRatio = 0



                if math.fabs(nextBandWidthRatio) > maxBandwidthRatio:
                    ng0_index = j
                    # print ng0_index # sanity check
                    maxBandwidthRatio = math.fabs(nextBandWidthRatio)

        # else:
            #invalid band structure

         # ng0_index determined


    if ng0_index == 0:
        # print "\nInsufficient Group Index for solution"
        # return worst case dummy solutions (will be ignored by SPEA
        # and scored badly by Gradient/Differential Evolutions algs
        output_map = {}
        output_map["bandwidth"] = 0
        output_map["ng0"] = 0
        output_map["avgLoss"] = 100000
        output_map["GBP"] = 0
        output_map["loss_at_ng0"]= 100000
        output_map["delay"] = 0
        return output_map

    else:

        ng0 = groupIndexMap[ng0_index]

        potentialBandwidthIndexes = []

        for i in range(1,experiment.kinterp + 1):
            if (((groupIndexMap[i] < 1.1*ng0) and (groupIndexMap[i] > 0.9*ng0)) or ((groupIndexMap[i] > 1.1*ng0) and (groupIndexMap[i] < 0.9*ng0))) and lossContrastMap[i] < 0.1*lossMap[i]:
                potentialBandwidthIndexes.append(i)

        # print "\nPotential bandwidths"
        # print potentialBandwidthIndexes # sanity check
        # evaluate potentialBandwidthIndexes as a 'continuous' set of values
        bandwidthMinIndex = min(potentialBandwidthIndexes)
        bandwidthMaxIndex = max(potentialBandwidthIndexes)

        # remove disjoint bandwidths below ng0 from potentialBandwidthIndexes
        for m in range(bandwidthMinIndex, ng0_index):
            if m not in potentialBandwidthIndexes:
                for n in range(bandwidthMinIndex,m):
                    if n in potentialBandwidthIndexes:
                        potentialBandwidthIndexes.remove(n)

        # remove disjoint bandwidths above ng0 from potentialBandwidthIndexes
        startRemoving = False
        for k in range(ng0_index, bandwidthMaxIndex + 1):
            if startRemoving:
                if k in potentialBandwidthIndexes:
                        potentialBandwidthIndexes.remove(k)
            else:
                if k not in potentialBandwidthIndexes:
                    startRemoving = True

        #print "DEBUG: Band Region: " + str(potentialBandwidthIndexes) # sanity check
        #print "DEBUG: Region GBP: " + str(maxBandwidthRatio)
        #print "DEBUG: ng0 index: " + str(ng0_index)
        #print "DEBUG ng0: " + str(ng0)
        #print "DEBUG: Loss: " + str(lossMap[ng0_index])
        # prepare to compute average loss and bandwidth

        bandwidthMinIndex = min(potentialBandwidthIndexes)
        bandwidthMaxIndex = max(potentialBandwidthIndexes)

        """
        # select group index that minimizes change in ng
        delta = math.fabs(ng0)
        for i in range(bandwidthMinIndex, bandwidthMaxIndex):
            delta_check = math.fabs(math.fabs(groupIndexMap[i + 1]) - math.fabs(groupIndexMap[i]))
            if delta_check < delta:
                delta = delta_check
                ng0_index = i

        """

        """

        # centralize group index in flat band then minimize group index based on loss
        ng0_index = (bandwidthMaxIndex + bandwidthMinIndex)/2
        extended_ng0_range = 0
        if (bandwidthMaxIndex - bandwidthMinIndex) != 0:
            extended_ng0_range = max(int(math.floor(math.log(bandwidthMaxIndex - bandwidthMinIndex))),0)
        min_index_loss_ratio = lossMap[ng0_index]/math.pow(ng0, 2)
        min_loss_index = ng0_index

        for b in range(ng0_index -extended_ng0_range, ng0_index + extended_ng0_range + 1):

            if b in potentialBandwidthIndexes:
                index_loss_ratio = math.fabs(lossMap[b])/math.pow(groupIndexMap[b], 2)
                if index_loss_ratio < min_index_loss_ratio:
                    min_index_loss_ratio = index_loss_ratio
                    min_loss_index = b

        ng0_index = min_loss_index
        """
        ng0 = groupIndexMap[ng0_index]


        potentialBandwidthIndexes = []

        for i in range(1,experiment.kinterp + 1):
            if (((groupIndexMap[i] < 1.1*ng0) and (groupIndexMap[i] > 0.9*ng0)) or ((groupIndexMap[i] > 1.1*ng0) and (groupIndexMap[i] < 0.9*ng0))) and lossContrastMap[i] < 0.1*lossMap[i]:
                potentialBandwidthIndexes.append(i)

        # print "\nPotential bandwidths"
        # print potentialBandwidthIndexes # sanity check
        # evaluate potentialBandwidthIndexes as a 'continuous' set of values
        bandwidthMinIndex = min(potentialBandwidthIndexes)
        bandwidthMaxIndex = max(potentialBandwidthIndexes)

        # remove disjoint bandwidths below ng0 from potentialBandwidthIndexes
        for m in range(bandwidthMinIndex, ng0_index):
            if m not in potentialBandwidthIndexes:
                for n in range(bandwidthMinIndex,m):
                    if n in potentialBandwidthIndexes:
                        potentialBandwidthIndexes.remove(n)

        # remove disjoint bandwidths above ng0 from potentialBandwidthIndexes
        startRemoving = False
        for k in range(ng0_index, bandwidthMaxIndex + 1):
            if startRemoving:
                if k in potentialBandwidthIndexes:
                        potentialBandwidthIndexes.remove(k)
            else:
                if k not in potentialBandwidthIndexes:
                    startRemoving = True

        print "DEBUG: Central Band Region: " + str(potentialBandwidthIndexes) # sanity check
        # prepare to compute average loss and bandwidth

        print "DEBUG: ngo index: " + str(ng0_index)
        #print "\n" + "ngo: " + str(ng0)
         # calculate loss at group index
        loss_at_ng0 = lossMap[ng0_index]

        # stores the (te)frequency at viable k points
        viableBandwidths = {}

        # stores the group index values at viable k points
        viableGroupIndexes = {}

        # computes loss at the viable k points
        viableLosses = {}


        for p in potentialBandwidthIndexes:
            viableBandwidths[p] = bandwidths[p]
            viableGroupIndexes[p] = math.fabs(groupIndexMap[p])
            viableLosses[p] = lossMap[p]

        # print viableBandwidths # sanity check
        # bandwidthNormalized is essentially dw/w0 for ng_i > 0.9*ng0, ng_i < 1.1*ng0
        bandwidthNormalized = ( max(viableBandwidths.values())
                                - min(viableBandwidths.values()) ) / viableBandwidths[ng0_index]


        # compute the average group index (this may not be a useful measure)
        avgGroupIndex = sum(viableGroupIndexes.values()) / float(len(viableGroupIndexes.values()))
        # avgGroupIndex = sum(groupIndexMap.values()) / float(len(groupIndexMap.values()))
        # print lossMap

        avgLoss = sum(viableLosses.values()) / float(len(viableLosses.values()))
        # avgLoss = sum(lossMap.values()) / float(len(lossMap.values()))

        # compute BGP
        bandWidthRatio = avgGroupIndex*bandwidthNormalized # could use ng0
        # bandwidthRatio = ng0*bandwidthNormalized

        length = 120

        loss_per_delay = lossMap[ng0_index] / (length * ng0)

        # compute GVD using finite differences
        # uses central finite difference if possible,
        # if not, evaluates GVD with the suitable forward/backward finite difference
        if (ng0_index + 1) in potentialBandwidthIndexes:
            ng0_plus = viableGroupIndexes[ng0_index + 1]
            omega_plus = viableBandwidths[ng0_index + 1]
        else:
            ng0_plus = ng0
            omega_plus = viableBandwidths[ng0_index]

        if (ng0_index - 1) in potentialBandwidthIndexes:
            ng0_minus = viableGroupIndexes[ng0_index - 1]
            omega_minus = viableBandwidths[ng0_index - 1]
        else:
            ng0_minus = ng0
            omega_minus = viableBandwidths[ng0_index]

        c_over_a = float(300000000) / float(410)
        delta_omega = math.fabs(omega_plus - omega_minus) * c_over_a # c_over_a changes delta_omega from units of (c/a) to units of 1/ns
        delta_ng0 = math.fabs(ng0_plus - ng0_minus)
        c = 30 # speed of light

        if delta_omega > 0:
            group_velocity_dispersion = delta_ng0 / (c * delta_omega) # in (ns^2)/cm
        else:
            group_velocity_dispersion = 100000
        # end of GVD computation

        #print "[Debug] GVD: " + str(group_velocity_dispersion) # placeholder until GVD integration

        #
        maximum_acceptable_loss = 10
        length_loss_limited = float(maximum_acceptable_loss) / loss_at_ng0 # in dB/ dB/ cm
        delay_loss_limited = (math.fabs(ng0)-5) * length_loss_limited / c * 1000 # in ps

        #print "[Debug] Loss Delay: " + str(delay_loss_limited)

        # delay formula is derived in respect to "Dispersion engineered slow light in photonic crystals: a comparison" by Sebastian Schulz et al
        initial_pulse_width = 0.045 # t_0 (in ns)
        length_gvd_limited = initial_pulse_width**2 / (4 * math.log(2) * group_velocity_dispersion) # in units of ns^2/ ns^2/cm
        delay_gvd_limited = (math.fabs(ng0)-5) * math.fabs(length_gvd_limited) / c * 1000 # in ps

        #print "[Debug] GVD Delay: " + str(delay_gvd_limited) + "\n"

        if delay_gvd_limited < delay_loss_limited:
            delay = delay_gvd_limited
            print "DEBUG: GVD limited structure"
            print "DEBUG: Length: " + str(length_gvd_limited * 10) + " mm"
        else:
            delay = delay_loss_limited
            print "DEBUG: Loss limited structure"
            print "DEBUG: Length: " + str(length_loss_limited * 10) + " mm"


        #delay = min(delay_loss_limited, delay_gvd_limited)

        #ngo = avgGroupIndex

        output_map ={}

        output_map["bandwidth"] = float("{0:.4f}".format(bandwidthNormalized))
        output_map["ng0"] = float("{0:.4f}".format(-ng0))
        output_map["avgLoss"] = float("{0:.4f}".format(avgLoss))
        output_map["GBP"] = float("{0:.4f}".format(bandWidthRatio))
        output_map["loss_at_ng0"]= float("{0:.4f}".format(loss_at_ng0))
        output_map["delay"] = float("{0:.4f}".format(delay))

        return output_map


