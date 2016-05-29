# Sean Billings, 2015

import constraints

# An experiment is a class representation for an mpb command line run
# by default the mpb run is split across 6 cores, but this can be augments
# and no parameter adjustments
class PhCWDesign(object):
        # initializes the Design with experiment and a set of solutions

        def __init__(self, solution_vector, score, constraints):

            self.solution_vector = solution_vector
            self.figures_of_merit = {}
            self.score = score
            self.constraints= constraints

        def constrain(self):
            constraints.fix(self.solution_vector, self.constraints)

        def set_objectives(self, new_objectives):
            self.figures_of_merit = new_objectives

        @property
        def copy_phc(self):
            new_phc = PhCWDesign(self.solution_vector.copy(), self.score, self.constraints)
            new_phc.set_objectives(self.figures_of_merit.copy())
            return new_phc



