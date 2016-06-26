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

class MonteCarloOptimizer(Optimizer):
    __metaclass__ = ABCMeta



    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.solutions = []

    def optimize(self, population, max_iterations, fault_tolerance, step_size):
        solutions = self.monte_carlo(population, max_iterations, fault_tolerance, step_size)

        return solutions

    def monte_carlo(self, population, max_iterations, fault_tolerance, step_size):
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

                if result.score < check_pcw.score:
                    print "\nIteration " + str(i) + " of " + str(max_iterations) + " results"
                    print result.solution_vector
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
                    print check_pcw.solution_vector
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

    def explore_path(self, pcw, step_size):
        descent_direction = random.sample(pcw.solution_vector, 1)[0]
        print "Step Direction: " + str(descent_direction)
        if random.Random() > 0.5:
             descent_magnitude = 1
        else:
            descent_magnitude = -1
        print "Step Magnitude: " + str(descent_magnitude*step_size)
        # update next_pcw
        next_pcw = pcw.copy_phc
        next_pcw.solution_vector[descent_direction] -= descent_magnitude*step_size
        next_pcw.constrain()
        self.objective_function.evaluate(next_pcw)

        return next_pcw
