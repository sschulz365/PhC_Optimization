import random
import numpy
import constraints
import copy
import mpbParser
import math
from optimizer import Optimizer
from abc import ABCMeta, abstractmethod
from objectiveFunctions import ObjectiveFunction
from photonicCrystalDesign import PhCWDesign

# minimizes the score a set of pcw solution vectors using the designated objectiveFunction
# according to a set of stopping conditions
# and returns the optimized vectors as a set of pcw solutions

class StochasticGradientDescentOptimizer(Optimizer):
    __metaclass__ = ABCMeta



    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.solutions = []

    def optimize(self, population, max_iterations, fault_tolerance):
        solutions = self.stochastic_gradient_descent(population, max_iterations, fault_tolerance)

        return solutions

    def stochastic_gradient_descent(self, population, max_iterations, fault_tolerance):
        solutions = []
        j = 1
        # optimize vectors with gradient descent
        for pcw in population:
            vector = pcw.solution_vector
            print "\nPerforming Stochastic Gradient Descent on " + str(vector)
            print "\nVector " + str(j) + " of " + str(len(population))
            j += 1
            # print vector
            self.objective_function.evaluate(pcw)
            original_fom = str(pcw.figures_of_merit)
            print "\nInitial score: " + str(pcw.score)
            i = 1
            check_pcw = pcw.copy_phc
            bad_direction_count = 0
            while i <= max_iterations:
                print "\n\nIteration: " + str(i)
                # the Stochastic Gradient Descent randomly improves 'vector' until max_iterations is met
                # (see above for method specifications)


                result = self.partial_gradient_descent(check_pcw)
                #vector = check_pcw.solution_vector

                if result.score <= check_pcw.score:
                    print "\nIiteration " + str(i) + " of " + str(max_iterations) + " results"
                    print vector
                    print "\nScore: " + str(result.score)
                    print "Total Improvement: " + str(pcw.score - result.score)
                    print "Original FOM: " + original_fom
                    print "New FOM: " + str(result.figures_of_merit)
                    check_pcw = result
                    bad_direction_count = 0



                else:
                    bad_direction_count += 1


                if bad_direction_count >= fault_tolerance:
                    i = max_iterations
                    print "Reached fault_limit"
                    print vector
                    print "\nFinal Score: " + str(check_pcw.score)
                    print "Total Improvement: " + str(pcw.score - check_pcw.score)
                    print "Original FOM: " + original_fom
                    print "New FOM: " + str(result.figures_of_merit)
                    check_pcw = result

                i += 1

            # store our vector in a set of solutions
            solutions.append(check_pcw)

        # end for vector
        return solutions

    def partial_gradient_descent(self, pcw):
        descent_direction =  random.sample(pcw.solution_vector, 1)

        delta = 0.0001

        pcw_with_vector_plus = pcw.copy_phc
        pcw_with_vector_minus = pcw.copy_phc

        pcw_with_vector_plus.solution_vector[descent_direction] = pcw_with_vector_plus.solution_vector[descent_direction] + delta
        pcw_with_vector_minus.solution_vector[descent_direction] = pcw_with_vector_plus.solution_vector[descent_direction] - delta

        self.objective_function.evaluate(pcw_with_vector_plus)
        self.objective_function.evaluate(pcw_with_vector_minus)

        delta_plus_score = pcw_with_vector_plus.score
        delta_minus_score = pcw_with_vector_minus.score

        # Below is the first order partial derivative in key
        # computed by central finite difference scaled by delta

        descent_magnitude = float("{0:.6f}".format((delta_plus_score - delta_minus_score) / 2))

        # update next_pcw
        next_pcw = pcw.copy_phc
        next_pcw.solution_vector[descent_direction] -= descent_magnitude
        self.objective_function.evaluate(next_pcw)

        return next_pcw