#Sean Billings, 2015
# objectiveFunctions
from mpbParser import parseObjFunctionParams, parseObjFunctionParams3D
import math
import os
from abc import ABCMeta, abstractmethod

# Extendible class for Objective Function
# Extensions of this class are passes to the optimizers
class ObjectiveFunction:
    __metaclass__ = ABCMeta

    # constructor parameters
    def __init_(self, experiment, objectives):
        self.experiment = experiment
        self.objectives = objectives

    # determines the mpb command that will be used
    # for future objective function calls
    @abstractmethod
    def set_experiment(self, experiment):
        self.experiment = experiment

    # returns a collection of scores/objectives
    @abstractmethod
    def evaluate(self, solution): pass




# Class implementation of the weighted objective function for 2D simulations
class WeightedSumObjectiveFunction(ObjectiveFunction):

    # Extended constructor
    def __init__(self, weights, experiment_obj):
        self.experiment = experiment_obj
        self.weights = weights

    # determines the mpb command that will be used
    # for future objective function calls
    def set_experiment(self, experiment_obj):
        super(experiment_obj)

   # The following implements a weighted sum scoring function for optimization routines
    # It reduces a multi-criteria optimization problem, to a single-value optimization problem
    # This method executes a command line execution of mpb, and then parses (via mpbParser.py) the output text
    # a reference to the output text is found in self.experiment
    def evaluate(self, pcw):
        #assert isinstance(pcw,PhCWDesign)

        self.experiment.setParams(pcw.solution_vector)
        # currently using calc type 4 is required
        # experiment.setCalculationType(4)
        self.experiment.perform()

        # parse the objParams from the experiment
        # see mpbParser.py for the definition of this set of values
        # (including parsing failure conditions)
        results = parseObjFunctionParams(self.experiment)


        # in the case where bandwidth/bgp is undetermined/ too small to approximate
        # we replace bandwidth/bgp with a tiny value, so that we do not divide by 0 in our objective function.
        if results["bandwidth"] < 0.00000001:
            results["bandwidth"] = 0.00000001

        if math.fabs(results["GBP"]) < 0.00000001:
            results["GBP"] = 0.00000001

        if math.fabs(results["ng0"]) < 0.00000001:
            results["ng0"] = 0.00000001

        if results["delay"] < 0.00000001:
            results["delay"] = 0.00000001

        bandwidth = results["bandwidth"]
        ng0 = results["ng0"]
        avgLoss = results["avgLoss"]
        gbp = results["GBP"]
        loss_at_ng0 = results["loss_at_ng0"]
        delay = results["delay"]

        pcw.figures_of_merit = results

        # the weights list is defined in the execution of runExperiment.py / runOptimizer.py
        # weight for bandwidth
        w1 = self.weights[0]
        # weight for group index
        w2 = self.weights[1]
        # weight for average loss
        w3 = self.weights[2]
        # weight for bandwidth-group_index product
        w4 = self.weights[3]
        # weight for loss at ng0
        w5 = self.weights[4]
        # weight for delay
        w6 = self.weights[5]


        # evaluate weighted sum objected function and return
        pcw.score = float("{0:.4f}".format(math.sqrt((w1/bandwidth)**2 + (w2/ng0)**2 + (w3*avgLoss)**2 + ((w5*loss_at_ng0)**2) + (w4/gbp)**2 + (w6/delay)**2)))

        self.store_solution(pcw)


    def store_solution(self, pcw):
        paramMap = pcw.solution_vector
        results = pcw.figures_of_merit
        with open(os.getcwd() + "/pcw_storage.txt","a") as out_file:
            out_file.write("\n%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"
                %(
                    paramMap["r0"],
                    paramMap["r1"],
                    paramMap["r2"],
                    paramMap["r3"],
                    paramMap["s1"],
                    paramMap["s2"],
                    paramMap["s3"],
                    paramMap["p1"],
                    paramMap["p2"],
                    paramMap["p3"],
                    results["bandwidth"],
                    results["ng0"],
                    results["GBP"],
                    results["avgLoss"],
                    results["loss_at_ng0"],
                    results["delay"]
            )
            )



# Class implementation of the weighted objective function for 3D simulations
# note there is no inclusion of the laplacian (fabrication stability) operator
# because it would take hours to compute
class WeightedSumObjectiveFunction3D(ObjectiveFunction):

    def __init__(self, weights, experiment):
        self.experiment = experiment
        self.weights = weights

    def set_experiment(self, experiment):
        super(experiment)

    # The following implements a weighted sum scoring function for optimization routines
    # It reduces a multi-criteria optimization problem, to a single-value optimization problem
    # This method executes a command line execution of mpb, and then parses (via mpbParser.py) the output text
    # a reference to the output text is found in self.experiment
    def evaluate(self, pcw):


        self.experiment.setParams(pcw.solution_vector)
        # currently using calc type 4 is required
        # experiment.setCalculationType(4)
        self.experiment.perform()

        # parse the results from the experiment
        # see mpbParser.py for the definition of this set of values
        # (including parsing failure conditions)
        results = parseObjFunctionParams3D(self.experiment)

        if results == 0:
            print "Parsing failure"
            return [float(1000000000000)]
        # the source of parsing failures is still undetermined
        else:

            # in the case where bandwidth/bgp is undetermined/ too small to approximate
            # we replace bandwidth/bgp with a tiny value, so that we do not divide by 0 in our objective function.
            if results["bandwidth"] < 0.00000001:
                results["bandwidth"] = 0.00000001

            if math.fabs(results["bgp"]) < 0.00000001:
                results["GBP"] = 0.00000001

            if math.fabs(results["ng0"]) < 0.00000001:
                results["ng0"] = 0.00000001

            if results["delay"] < 0.00000001:
                results["delay"] = 0.00000001

            bandwidth = results["bandwidth"]
            ng0 = results["ng0"]
            avgLoss = results["avgLoss"]
            gbp = results["GBP"]
            loss_at_ng0 = results["loss_at_ng0"]
            delay = results["delay"]

            pcw.figures_of_merit = results

            # the weights list is defined in the execution of runExperiment.py / runOptimizer.py
            # weight for bandwidth
            w1 = self.weights[0]
            # weight for group index
            w2 = self.weights[1]
            # weight for average loss
            w3 = self.weights[2]
            # weight for bandwidth-group_index product
            w4 = self.weights[3]
            # weight for loss at ng0
            w5 = self.weights[4]
            # weight for delay
            w6 = self.weights[5]


            # evaluate weighted sum objected function and return
            pcw.score = float("{0:.4f}".format(math.sqrt((w1/bandwidth)**2 + (w2/ng0)**2 + (w3*avgLoss)**2 + ((w5*loss_at_ng0)**2) + (w4/gbp)**2 + (w6/delay)**2)))



#TODO: update with pcw class formalism
class IdealDifferentialObjectiveFunction(ObjectiveFunction):

    def __init__(self, weights, experiment, ideal_solution):
        self.experiment = experiment
        self.weights = weights
        self.ideal_solution = ideal_solution

    def set_experiment(self, experiment):
        super(experiment)

    def evaluate(self, solution):

        # the weights list is defined in the execution of runExperiment.py / runOptimizer.py
        # weight for bandwidth

        # Set w_x*ideal_x to 1
        w1 = self.weights[0]
        # weight for group index
        w2 = self.weights[1]
        # weight for average loss
        w3 = self.weights[2]
        # weight for bandwidth-group_index product
        w4 = self.weights[3]
        # weight for loss at ng0
        w5 = self.weights[4]
        # weight for delay
        w6 = self.weights[5]

        ideal_group_index = self.ideal_solution[0]
        ideal_bandwidth = self.ideal_solution[1]
        ideal_loss_at_group_index = self.ideal_solution[2]
        ideal_bgp = self.ideal_solution[3]
        ideal_delay = self.ideal_solution[4]


        self.experiment.setParams(solution)
        # currently using calc type 4 is required
        # experiment.setCalculationType(4)
        self.experiment.perform()

        # parse the objParams from the experiment
        # see mpbParser.py for the definition of this set of values
        # (including parsing failure conditions)
        #objParams = parseObjFunctionParams3D(self.experiment) # 3D
        objParams = parseObjFunctionParams(self.experiment)



        if objParams == 0:
             print "Parsing failure"
             return [float(1000000000000)]
        # the source of parsing failures is still undetermined
        else:
            bandwidth = float("{0:.4f}".format(objParams[0]))
            group_index = float("{0:.4f}".format(objParams[1]))
            avgLoss = float("{0:.4f}".format(objParams[2]))
            bgp = float("{0:.4f}".format(objParams[3]))
            loss_at_ng0 = float("{0:.4f}".format(objParams[4]))
            delay = float("{0:.4f}".format(objParams[5 ]))


         # ideal - score

        bandwidth_difference = (ideal_bandwidth - bandwidth)/ideal_bandwidth
        ng0_difference = (ideal_group_index - math.fabs(group_index))/ideal_group_index
        loss_difference = (loss_at_ng0 - ideal_loss_at_group_index) /ideal_loss_at_group_index # inverted because we want to minimize loss
        bgp_difference = (ideal_bgp - bgp)/ ideal_bgp
        delay_difference =(ideal_delay - delay)/delay
        score = float("{0:.7f}".format(w1*bandwidth_difference + w2*ng0_difference + w4*bgp_difference + w5*loss_difference + w6*delay_difference))
        
        # note score can be negative if the ideal is surpassed
        results = [ score, bandwidth, group_index, avgLoss, bgp, loss_at_ng0, delay]
        return results


