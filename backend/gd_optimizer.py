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
            check_pcw = pcw.copy_phc
            prev_pcw = check_pcw.copy_phc
            while i <= max_iterations:
                print "\n\nIteration: " + str(i)
                # the Gradient Descent method recursively improves 'vector' until a stopping condition is met
                # (see above for method specifications)

                result = self.gradientDescent(check_pcw,
                                         descent_scaler,
                                         completion_scaler,
                                         alpha_scaler)


                if check_pcw.score <= result.score:
                    i = max_iterations
                    # This results means that the solution is approximately convergent to a local minima
                    print "Divergent iteration results"
                    print vector
                    print "\nFinal Score: " + str(check_pcw.score)
                    print "Total Improvement: " + str(pcw.score - check_pcw.score)
                    print "Original FOM: " + original_fom
                    print "New FOM: " + str(result.figures_of_merit)
                    check_pcw = prev_pcw

                else:
                    print "\nIiteration " + str(i) + " of " + str(max_iterations) + " results"
                    print vector
                    print "\nScore: " + str(result.score)
                    print "Total Improvement: " + str(check_pcw.score - result.score)
                    print "Original FOM: " + original_fom
                    print "New FOM: " + str(result.figures_of_merit)
                    prev_pcw = check_pcw.copy_phc
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
        initial_score = pcw.score

        if initial_score > 10000: # arbitrary max score
            print "\nDescent not achievable"
            return pcw

        # create next_vector to be used in gradient descent
        next_pcw = pcw.copy_phc
        initial_pcw = pcw.copy_phc

        # the inner product is used to to verify that the wolfe conditions are satisfied
        gradient_innerProduct_terms = {}
        gradientValues = {}


        """
        # In the following we compute the partial derivatives
        # of the objective function, with respect to the parameters,
        # and store the terms as a representation of the gradient in gradientValues
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

            pcw_with_vector_plus = next_pcw.copy_phc
            pcw_with_vector_plus.solution_vector[key] = pcw_with_vector_plus.solution_vector[key] + delta
            self.objective_function.evaluate(pcw_with_vector_plus)

            deltaPlusScore = pcw_with_vector_plus.score


            # Below is the first order partial derivative in key
            # computed by forward finite difference

            gradientValues[key] = float("{0:.6f}".format( (deltaPlusScore - initial_score)  * ( 0.00001/delta)))

            # removing delta is incorrect, but it can help the scaling of score output
            # multiplying by 0.00001 helps scale the gradient into the constrained bounds of [0,1]
            # this initial scaling is handled during the augmented backtracking approach below


            # Below is the first order partial derivative in key
            # computed by central finite difference
            """
            gradientValues[key] = ((deltaPlusScore - deltaMinusScore) / 2*delta) * 0.0001
            """


            # update next_vector
            next_vector[key] = vector[key] - gradientValues[key]
            #                                alpha_scaler*

            # update gradient_innerProduct_terms
            gradient_innerProduct_terms[key] = (-1)*(gradientValues[key]**2)

        # end gradient calculation

        print "Gradient Computed" # + str(gradientValues)
        print "Gradient: " + str(gradientValues)


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

        if gradient_innerProduct > -0.000001:
            return initial_pcw

        #the gradient factor is defined by Grad(f_k) transpose applied to p_k
        # it is therefore the gradient inner product because p_k is defined as - grad(f_k)
        gradient_factor = completion_scaler*alpha_scaler*gradient_innerProduct
        print "Gradient Factor: " + str(gradient_factor)

        # THE FOLLOWING COULD BE IMPROVED WITH THE FORMALISM OF
        # NONLINEAR CONDITIONAL OPTIMIZATION
        attempts = 0
        print "\nScaling Descent\n"
        print "Score: " + str(next_score)

        #gradient factor is negative
        descent_score = initial_score
        while next_score > (descent_score + gradient_factor) and attempts < 4:


            # recompute next_vector using the original vector
            # and the alpha adjusted gradient where p_k = -a*GRAD(x_k)
            for key in vector.keys():
               next_vector[key] = (vector[key] - alpha_scaler*gradientValues[key])

            next_pcw.solution_vector = next_vector
            next_pcw.constrain() # may want to use lagrangian methods instead
            self.objective_function.evaluate(next_pcw)
            next_score = next_pcw.score


            print "\n" + str(alpha_scaler)
            print next_vector
            print "Score: " + str(next_score)
            alpha_scaler = descent_scaler*alpha_scaler

            gradient_factor = completion_scaler*alpha_scaler*gradient_innerProduct

            attempts+= 1

        if attempts > 4:
            print "\nDescent not achievable"
            return initial_pcw

        if next_score < (descent_score + gradient_factor):
            return next_pcw

        #TODO: revisions to below
        # if a vectors score already satisfies the wolfe conditions,
        # then we scale up the value of the gradient descent
        if attempts == 0:
            last_score = initial_score
            last_ascent_pcw = next_pcw.copy_phc

            print "Descent is Relatively Steep!\n"
            # scale  the gradient at the values 10, 100, 1000, 10000
            # where 10000 * gradientValues[key] is the actual gradient value
            # in contrast to the reduced (scaled by 0.0001) gradient that is computed
            attempts = 1
            ascent_scaler = 5
            ascent_pcw = next_pcw.copy_phc
            ascent_vector = ascent_pcw.solution_vector

            for key in vector.keys():
                ascent_vector[key] = (vector[key] - (ascent_scaler)*gradientValues[key])

            ascent_pcw.solution_vector = ascent_vector
            ascent_pcw.constrain()
            self.objective_function.evaluate(ascent_pcw)
            ascent_score = ascent_pcw.score
            gradient_factor = completion_scaler*ascent_scaler*gradient_innerProduct

            if ascent_pcw.score > last_score:
                    return last_ascent_pcw
            last_ascent_pcw = ascent_pcw.copy_phc

            print "Scaling Ascent"
            print "\nAscent Scaler: " + str(ascent_scaler)
            print ascent_vector
            print "Score: " + str(ascent_score)
            ascent_scaler = 5*ascent_scaler

            while ascent_score < (last_score + gradient_factor) and attempts < 5:
                last_score = ascent_score
                ascent_vector = ascent_pcw.solution_vector
                for key in vector.keys():
                    # assignment and boundary conditioning
                    ascent_vector[key] = min( [ 1, (next_vector[key] - (ascent_scaler)*gradientValues[key])])
                    # more on the fly boundary conditioning
                    ascent_vector[key] = max( [-1, ascent_vector[key]])


                ascent_pcw.solution_vector = ascent_vector
                ascent_pcw.constrain()
                self.objective_function.evaluate(ascent_pcw)

                print "\n" + str(ascent_scaler)
                print ascent_vector
                print "Score: " + str(ascent_score)
                ascent_scaler = 5*ascent_scaler
                attempts+= 1

                gradient_factor = completion_scaler*ascent_scaler*gradient_innerProduct

                if ascent_pcw.score >= last_score:
                    return last_ascent_pcw

                last_ascent_pcw = ascent_pcw.copy_phc
                ascent_score = ascent_pcw.score

            return last_ascent_pcw

        return initial_pcw
