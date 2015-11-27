__author__ = 'sean'
import constraints

# An experiment is a class representation for an mpb command line run
# by default the mpb run is split across 6 cores, but this can be augments
# and no parameter adjustments
class PhCWDesign(object):
        # initializes the Design with experiment and a set of solutions

        def __init__(self, solution_vector, score, constraints):
            """
            :param experiment:
            :param solution_vector:
            :param figures_of_merit:
            :param score:
            :param parser: callable script for parsing objective
            :param constraints:
            :return:
            """

            self.solution_vector = solution_vector
            self.figures_of_merit = {}
            self.score = score
            self.constraints= constraints

        def constrain(self):
            constraints.fix(self.solution_vector, self.constraints)

        def set_objectives(self, new_objectives):
            self.figures_of_merit = new_objectives

        def copy_phc(self):
            new_phc = PhCWDesign(self.solution_vector, self.score, self.constraints)
            new_phc.set_objectives(self.figures_of_merit)
            return new_phc



