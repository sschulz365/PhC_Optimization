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

    def optimize(self, population, max_iterations):
        solutions = self.stochastic_gradient_descent(population, max_iterations)

        return solutions

    def stochastic_gradient_descent(self, population, max_iterations):
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
            # could keep track of scores
            i = 1
            check_pcw = pcw.copy_phc
            while i <= max_iterations:
                print "\n\nIteration: " + str(i)
                # the Gradient Descent method recursively improves 'vector' until a stopping condition is met
                # (see above for method specifications)

                result = self.gradientDescent(check_pcw,
                                         descent_scaler,
                                         completion_scaler,
                                         alpha_scaler)

                #vector = check_pcw.solution_vector

                if check_pcw.score <= result.score:
                    i = max_iterations
                    # This results means that the solution is approximately convergent to a local minima
                    print "Divergent iteration results"
                    print vector
                    print "\nFinal Score: " + str(check_pcw.score)
                    print "Total Improvement: " + str(pcw.score - check_pcw.score)
                    print "Original FOM: " + original_fom
                    print "New FOM: " + str(result.figures_of_merit)
                    check_pcw = result

                else:
                    print "\nIiteration " + str(i) + " of " + str(max_iterations) + " results"
                    print vector
                    print "\nScore: " + str(result.score)
                    print "Total Improvement: " + str(check_pcw.score - result.score)
                    print "Original FOM: " + original_fom
                    print "New FOM: " + str(result.figures_of_merit)
                    check_pcw = result

                i += 1

            # store our vector in a set of solutions
            solutions.append(check_pcw)

        # end for vector
        return solutions
