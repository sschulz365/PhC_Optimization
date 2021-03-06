#Sean Billings, 2015

import subprocess
import os
import mpbParser

# An experiment is a class representation for an mpb command line run
# by default the mpb run is split across 6 cores, but this can be augments

# should be refactored as W1Experiment
# and no parameter adjustments
class W1Experiment(object):

        # initializes the W1Experiment with default parameters (acts as a constructor specification)
        def __init__(self, mpb, inputFile, outputFile, parserIn=mpbParser, parseStrat="MAXGBP"):
            """

            :param mpb: command line path for the mpb program
            :param inputFile: specifies the structure of the waveguide
            :param outputFile: records output such as tefreqs, loss
            """
            self.inputFile = inputFile
            self.mpb = mpb
            self.outputFile = outputFile
            self.paramVectorString = ""
            self.calculationType = " calculation-type=4"
            self.c1 = 4 # constant for out of plane loss
            self.c2 = 220 # constant for backscattering loss
            self.fieldFraction= 0.827
            self.band = 23
            self.ks = 0.3
            self.ke = 0.5
            self.kinterp = 39
            self.split = "-split 6"
            self.dim3 = False
            self.parser=parserIn
            self.parseStrategy=parseStrat

        # execute the current instantiation of W1Experiment as a mpb calculation via the command line
        def perform(self):

            FNULL = open(os.devnull, 'w')
            subprocess.call(self.mpb + self.split + " Ks="+ str(self.ks) + " Ke=" + str(self.ke) + " Kinterp=" + str(self.kinterp) + self.calculationType + self.paramVectorString + ' %s > %s' %(self.inputFile, self.outputFile),
                            shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

        # set the parameters (as coded in the ctl with the format define-param) to values according to paramMap
        # paramMap example
        # {'p2': 0.07646386156785503, 'p3': 0.043493821552317305, 'p1': 0.07235772695508993,
        # 'r1': 0.017137050399521098, 'r2': 0.2053715543744835, 'r3': 0.03296599949382803,
        # 's3': 0.011459725712917944, 's2': 0.09901968783066335, 's1': 0.17175319110067366}
        def setParams(self, paramMap):
                self.paramVectorString = ""
                for x in paramMap.keys():
                        if 'r' in x or 'a' in x: 
                                self.paramVectorString += " " + x + "=" + str(paramMap[x])
                        else:
                                self.paramVectorString += " " + x + "=" + str(paramMap[x])
                        
                                

        # adjusts the calculation type (according to the type specifications as in W1_2D_v03.ctl)
        def setCalculationType(self, calcType):
            self.calculationType = " calculation-type=" + str(calcType)

        # a utility function to quickly calculate the Group Index, average Loss, BGP, etc
        # from the experiment instantiation
        # see mpbParser.py for output list format
        def extractFunctionParams(self):
            self.perform()
            if self.dim3:
                return self.parser.parseObjFunctionParams3D(self, self.parseStrategy)

            return self.parser.parseObjFunctionParams(self, self.parseStrategy)

        # determine which band we are considering for a given photonic crystal
        def setBand(self, newBand):
            self.band = newBand

        def setSplit(self, split_num):
            self.split = "-split " + str(split_num)

        def noSplit(self):
            self.split = ""
