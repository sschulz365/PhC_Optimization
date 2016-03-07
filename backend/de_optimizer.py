# Sean Billings, 2015

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

class DeOptimizer(Optimizer):
    __metaclass__ = ABCMeta



    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.solutions = []


    def optimize(self, population, max_generation, random_update_probability, elite_set_size):
        solutions = self.differentialEvolution(population, max_generation, random_update_probability,
                                               elite_set_size, self.objective_function)
        return solutions



    def differentialEvolution(self, population, max_generation, random_update_probability,
                              elite_set_size, objective_function):
        """ Differential Evolution Optimizer
        Creates and returns a set of solutions which minimize the passed objectiveFunction
        The results are not guaranteed local or global minima, but the DE algorithm
        provides good coverage of the solution space, and can provide great results
        """
        assert isinstance(objective_function, ObjectiveFunction)
        elites = {} # hashmap of scores and vectors
        top_scores = []
        #score population
        population_score = {} # hashmap of population indexes with scores
        population_size = len(population)
        for i in range(0, population_size):
            assert isinstance(population[i], PhCWDesign)

            print "ID: " + str(i)
            print population[i].solution_vector

            objective_function.evaluate(population[i])
            population_score[i] = population[i].score
            results = population[i].figures_of_merit
            solution_score = population[i].score

            #DEBUG

            print "\nFigures of merit: " + str(results)
            print "\nScore: " + str(solution_score)
            # each step will require a simulatation in MPB
            # print "Score: " + str(population_score[i])

            # supplement elites
            # this is buggy
            if i < elite_set_size:
                elites[population_score[i]] = population[i]
                top_scores.append(population_score[i])

        # iteratively update the population for each generation

        for generation in range(1, max_generation + 1):
            print "\nGeneration: " + str(generation)

            # update vectors (performs a standard DE mutation operator)
            print "\nUpdating Population..."
            for j in range(0, population_size):

                # generate random vectors (indices) xi from elites
                x1 = numpy.random.randint(0, elite_set_size)
                x2 = numpy.random.randint(0, elite_set_size)
                x3 = numpy.random.randint(0, elite_set_size)
                # could base on scoring metric


                # could clean this up to be more efficient
                while x1 == x2 or x2 == x3 or x1 == x3 or j == x1 or j == x2 or j == x3:
                    x1 = numpy.random.randint(1, elite_set_size)
                    x2 = numpy.random.randint(1, elite_set_size)
                    x3 = numpy.random.randint(1, elite_set_size)

                # Select vector from population
                u = population[j].copy_phc()

                vector1 = population[x1].solution_vector
                vector2 = population[x2].solution_vector
                vector3 = population[x3].solution_vector

                # Update entries in vector via differential
                for k in population[j].solution_vector.keys():
                    if random.random() < random_update_probability:
                        # scaling method
                        # u[i] = math.sqrt(((population[x3])[i])*(population[x1])[i])

                        # vector addition method
                        u.solution_vector[k] = vector3[k] + (vector1[k] - vector2[k])

                # fix is from the constraints library
                # it will map parameters in u back to acceptable bounds
                u.constrain()
                u_score = objective_function.evaluate(u)

                # determine whether to update the vector at j with the new vector u
                if population_score[j] > u_score:
                    population[j] = u.copy_phc()
                    population_score[j] = u_score

            #end for j

            print "\nGenerating Elites..."
            # There is a bug that involves not being able to fill the elites bag
            # to maximum capacity in the elite update/generate code below

            # compute the worst current elite score (max_score)
            max_score = 0
            for scores in top_scores:
                if scores > max_score:
                    max_score = scores

            # update elites with best new solutions from the population
            for k in range(0, population_size):
                if population_score[k] <= max_score:
                    elites[population_score[k]] = population[k].copy_phc()
                    top_scores.append(population_score[k])

            ## generate new elites

            # tricky set conversion to create unique, sorted version of top_scores
            unique_top_scores = list()
            top_scores = sorted(top_scores, key=lambda score: score)
            map(lambda x: not x in unique_top_scores and unique_top_scores.append(x), top_scores)

            top_scores = unique_top_scores

            #top_score_indexes = numpy.unique(top_scores)  #numpy.unique returns the sorted unique elements from top_scores
            print "\nTop Scores: " + str(top_scores)


            nextgen_elites = {}
            nextgen_top_scores = []

            # works because top_scores is sorted
            for i in range(0, elite_set_size):
                if i < len(top_scores):
                    nextgen_elites[top_scores[i]] = elites[top_scores[i]]
                    nextgen_top_scores.append(top_scores[i])

            elites = nextgen_elites.copy()
            top_scores = copy.deepcopy(nextgen_top_scores)

            print "\nElites: "
            for i in range(0, len(top_scores)):
                elite = elites[top_scores[i]]
                elite_vector = elite.solution_vector
                results = elite.figures_of_merit
                print "\nElite: " + str(elite_vector)
                print "\nResults: " + str(results)
                print "\nScore: " + str(elite.score)

                # end for  generation

        #final_elites = []
        #for i in range(0, elite_size):
        #    final_elites.append(elites[top_scores[i]])

        return elites