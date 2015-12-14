#Sean Billings, 2015

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

class GradientDescentOptimizer(Optimizer):
    __metaclass__ = ABCMeta



    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.solutions = []


    def optimize(self, population, descent_scaler, completion_scaler, alpha_scaler, max_iterations):
        solutions = self.gradientDescentAlgorithm(population, descent_scaler, completion_scaler,
                                                  alpha_scaler, max_iterations)
        return solutions


    def gradientDescentAlgorithm(self, population, descent_scaler, completion_scaler, alpha_scaler, max_iterations):

        solutions = []
        j = 1
        # optimize vectors with gradient descent
        for pcw in population:
            vector = pcw.solution_vector
            print "\nPerforming Gradient Descent on " + str(vector)
            print "\nVector " + str(j) + " of " + str(len(population))
            j += 1
            # print vector
            self.objective_function.evaluate(pcw)
            original_fom = str(pcw.figures_of_merit)
            print "\nInitial score: " + str(pcw.score)
            # could keep track of scores
            i = 1
            check_pcw = pcw.copy_phc()
            while i <= max_iterations:
                print "\n\nIteration: " + str(i)
                # the Gradient Descent method recursively improves 'vector' until a stopping condition is met
                # (see above for method specifications)

                result = self.gradientDescent(pcw,
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


    # Performs a version of the (steepest) gradient descent
    # with an augmented backtracking algorithm structure for a safe and efficient
    # gradient descent operation on 'vector'
    # returns a new vector which can be fed into a general gradient descent algorithm
    # This method is time consuming but the convergence is nice and it can approximate
    # local minima in the neighbourhood of the passed vector


    def gradientDescent(self, pcw, descent_scaler,
                        completion_scaler, alpha_scaler):

        #establish the base score to improve upon
        current_score = pcw.score

        if current_score > 10000: # arbitrary max score
            print "\nDescent not achievable"
            return pcw

        # create next_vector to be used in gradient descent
        next_pcw = pcw.copy_phc()
        default_pcw = pcw.copy_phc()

        # the inner product is used to to verify that the wolfe conditions are satisfied
        gradient_innerProduct_terms = {}
        gradientValues = {}

        """
        # the hessian can be used instead of the Identity matrix in computing the descent direction
        # more details can be found in Numerical Optimization : Nocedal, Wright In the section on line methods
        # in any case it would be far more efficient to approximate the hessian with the laplacian

        # hessianValues[key] = {}


        # In the following we compute the partial derivatives
        # of the objective function, with respect to the parameters,
        # and store the terms as a representation of the gradient in gradientValues

        # Code for calculating the Hessian is also commented below
        """
        vector = next_pcw.solution_vector
        next_vector = vector.copy()
        print "\nEvaluating gradient..."
        for key in vector.keys():
            print key + "..."

            # this value can be scaled
            delta = 0.00001

            # The following code implements:
            # -h finite difference for f(x - delta), where x is the current key
            """
            vectorMinusDeltaKey = vector.copy()
            vectorMinusDeltaKey[key] = vectorMinusDeltaKey[key] - delta
            constraints.fix(vectorMinusDeltaKey, constraintFunctions)
            deltaMinusScore = (objectiveFunction(vectorMinusDeltaKey, experiment))
            if deltaMinusScore > 10000:
                     deltaMinusScore = (objectiveFunction(vectorPlusDeltaKey, experiment))
            """
            # The following code implements:
            # +h finite difference for f(x + delta), where x is the current key
            vectorPlusDeltaKey = vector.copy()
            vectorPlusDeltaKey[key] = vectorPlusDeltaKey[key] + delta
            pcw_with_vector_plus = next_pcw.copy_phc()
            pcw_with_vector_plus.solution_vector[key] = pcw_with_vector_plus.solution_vector[key] + delta
            self.objective_function.evaluate(pcw_with_vector_plus)
            # constraints.fix(vectorPlusDeltaKey, constraintFunctions)
            deltaPlusScore = pcw_with_vector_plus.score


            # Below is the first order partial derivative in key
            # computed by forward finite difference

            gradientValues[key] = ((deltaPlusScore - current_score) / delta ) *0.00001

            # removing delta is incorrect, but it can help the scaling of score output
            # multiplying by 0.00001 helps scale the gradient into the constrained bounds of [0,1]
            # this initial scaling is dealt with during the augmented backtracking algorithm section


            # Below is the first order partial derivative in key
            # computed by central finite difference
            """
            gradientValues[key] = ((deltaPlusScore - deltaMinusScore) / 2*delta) * 0.0001
            """

            # second order approximated partial derivative in key (x)
            # approximated by 2nd order central
            # used to approximate Hessian (maybe with the Laplacian) potentially
            """
            f''(x) = f(x + delta) - 2f(x) + f(x-delta) / delta^2

            hessianValues[key] = (deltaPlusScore - 2*currentScore + deltaMinusScore) / delta**2
            """

            # print key + ": " + str(next_vector[key]) # sanity check

            # update next_vector
            next_vector[key] = vector[key] - gradientValues[key]
            #                                alpha_scaler*

            # print "Finite Difference: " + str(next_vector[key] - vector[key])

            # constraints.fix(next_vector, constraintFunctions)

            # update gradient_innerProduct_terms
            gradient_innerProduct_terms[key] = (-1)*(gradientValues[key]**2)

        # end gradient calculation

        print "\nGradient Computed" # + str(gradientValues)

        # print "Gradient and Hessian Computed" # in case of hessian calculation


        # under steepest desecent:
        # descent_direction = -deltaScores

        next_pcw.solution_vector = next_vector
        next_pcw.constrain()
        self.objective_function.evaluate(next_pcw)
        next_score = next_pcw.score

        # compute gradient_innerProduct

        gradient_innerProduct = 0
        for key in gradient_innerProduct_terms.keys():
            gradient_innerProduct += gradient_innerProduct_terms[key]

        #the gradient factor is defined by Grad(f_k) transpose applied to p_k
        # it is therefore the gradient inner product because p_k is defined as - grad(f_k)
        gradient_factor = completion_scaler*alpha_scaler*gradient_innerProduct
        print "\nGradient Factor: " + str(gradient_factor)


        # THE FOLLOWING COULD BE IMPROVED WITH THE FORMALISM OF
        # NONLINEAR CONDITIONAL OPTIMIZATION
        attempts = 0
        print "\nScaling Descent\n"
        while next_score > (current_score + gradient_factor) and attempts < 5:

            # recompute next_vector using the original vector
            # and the alpha adjusted gradient where p_k = -a*GRAD(x_k)
            for key in vector.keys():
               next_vector[key] = (vector[key] - alpha_scaler*gradientValues[key])

            next_pcw.solution_vector = next_vector
            next_pcw.constrain()
            self.objective_function.evaluate(next_pcw)
            next_score = next_pcw.score


            print "\n" + str(alpha_scaler)
            print next_vector
            alpha_scaler = descent_scaler*alpha_scaler

            gradient_factor = completion_scaler*alpha_scaler*gradient_innerProduct

            attempts+= 1

        if attempts > 4:
            print "\nDescent not achievable"
            return default_pcw

        #TODO: revisions to below
        # if a vectors score already satisfies the wolfe conditions,
        # then we scale up the value of the gradient descent
        if attempts == 0:
            # scale  the gradient at the values 10, 100, 1000, 10000
            # where 10000 * gradientValues[key] is the actual gradient value
            # in contrast to the reduced (scaled by 0.0001) gradient that is computed
            attempts = 1
            ascent_scaler = 10
            ascent_vector = next_pcw.solution_vector

            for key in vector.keys():
                ascent_vector[key] = (vector[key] - (ascent_scaler)*gradientValues[key])

            next_pcw.constrain()
            self.objective_function.evaluate(next_pcw)
            ascent_score = next_pcw.score
            gradient_factor = completion_scaler*ascent_scaler*gradient_innerProduct

            print "Maximizing Gradient"
            print "\n" + str(ascent_scaler)
            print ascent_vector
            ascent_scaler = 10*ascent_scaler

            while ascent_score > (current_score + gradient_factor) and attempts < 5:
                ascent_vector = next_vector.copy()
                for key in vector.keys():
                    ascent_vector[key] = min( [ 1, (next_vector[key] - (ascent_scaler)*gradientValues[key])])
                next_pcw.solution_vector = ascent_vector
                next_pcw.constrain()
                self.objective_function.evaluate(next_pcw)
                ascent_score = next_pcw.score

                print "Scaling Ascent"
                print "\n" + str(ascent_scaler)
                print ascent_vector
                ascent_scaler = 10*ascent_scaler
                attempts+= 1

                gradient_factor = completion_scaler*ascent_scaler*gradient_innerProduct

            if ascent_score < next_score:
                return next_pcw


        return next_pcw

        # Below is an implementation of the Cauchy Point method
        # for path choice of gradient descent

        """
        # gradient_L2norm = 0
        # for key in gradient_Values.keys():
        #     gradient_L2norm + = gradient_values[key]**2
        # gradient_L2norm = math.sqrt(gradient_L2norm)
        #
        # descent_vector = {}

        # for key in gradient_Values.keys():
        #     descent_vector[key] = (-1)*delta*gradient_Values[key] / gradient_L2norm
        #
        """


