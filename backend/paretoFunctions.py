# Sean Billings, 2015

from mpbParser import parseObjFunctionParams, parseObjFunctionParams3D
import math
from abc import ABCMeta, abstractmethod
from photonicCrystalDesign import PhCWDesign

class ParetoFunction:
    __metaclass__ = ABCMeta

    # constructor parameters
    def __init_(self, experiment):
        self.experiment = experiment

    # determines the mpb command that will be used
    # for future objective function calls
    def set_experiment(self, experiment):
        self.experiment = experiment

    @abstractmethod
    def evaluate(self, pcw): pass


    # returns a collection of scores/objectives
    @abstractmethod
    def dominates(self, solution_a, solution_b): pass


class ParetoMaxFunction(ParetoFunction):
    __metaclass__ = ABCMeta

    # constructor parameters
    def __init__(self, experiment, key_max_min_map):
        self.experiment = experiment
        self.key_list = key_max_min_map

    # determines the mpb command that will be used
    # for future objective function calls
    def set_experiment(self, experiment):
        super(experiment)

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

        considered_fom = {}

        for key in self.key_list.keys():
            if self.key_list[key] == "min":
                considered_fom[key] = 1/results[key]
            else:
                considered_fom[key] = results[key]
        pcw.figures_of_merit = considered_fom

        pcw.score = 0

    def set_keys(self, key_list):
        """
        Redefines the set of keys under consideration
        :param key_list:
        :return:
        """

        self.key_list = key_list

    # a and b are pcw objects
    # obj_function_results is the results from parseObjParams in mpbParser.py
    # parameter_map is defined in the runOptimizer.py module
    # both are dictionaries
    # this method simply evaluates whether a dominates b
    def dominates(self, a, b):
        #print "\n" + str(a)
        isinstance(a, PhCWDesign)

        a_fom = a.figures_of_merit
        b_fom = b.figures_of_merit

        for key in a.figures_of_merit.keys():
            if math.fabs(a_fom[key]) < math.fabs(b_fom[key]):
                return False

        return True



    # a and b are in the form [ obj_function_results, parameter_map]
    # obj_function_results is the results from parseObjParams in mpbParser.py
    # parameter_map is defined in the runOptimizer.py module
    # both are dictionaries
    # this method simply evaluates whether a strongly dombinates b
    def strongly_dominates(self, a, b):
        # print "\n" + str(a) + "dominates\n" + str(b) + "?\n"

        a_fom = a.figures_of_merit
        b_fom = b.figures_of_merit

        for key in a.figures_of_merit.keys():
            if math.fabs(a_fom[key]) <= math.fabs(b_fom[key]):
                return False

        return True