# Sean Billings, 2015

from abc import ABCMeta, abstractmethod
from objectiveFunctions import ObjectiveFunction
import math
import random
from photonicCrystalDesign import PhCWDesign
class Optimizer(object):


    def __init__(self, objective_function):
        #assert isinstance(objective_function, ObjectiveFunction)
        self.objective_function = objective_function


    @abstractmethod
    def optimize(self, population): pass
    """
    population is a set of PhCDesign
    returns a set of optimized PhCDesign
    """
    #

    @classmethod
    def createPopulation(self, population_size, pcw_archetype):
        """ creates a population which is a dictionary that maps parameters to generated random values
        see paramMap in runOptimizer.py for a population sample example
        """

        population = [pcw_archetype.copy_phc for i in range (0, population_size)]
        for i in range(0, population_size):
            assert isinstance(population[i], PhCWDesign)
            new_vector =  population[i].solution_vector.copy()
            for j in new_vector.keys():
                if 'r' in j:

                    new_vector[j] = (math.log(1-random.random())/-2)/5 + 0.2
                else:
                    scale = random.random() - 0.5
                    new_vector[j] = (math.log(1-random.random())/-10)*scale
                # exponetial distribution with lamda = 10

            population[i].solution_vector = new_vector
            population[i].constrain()


        return population

    @classmethod
    def set_objective_function(self,new_objective_function):
        self.objective_function = new_objective_function
        assert isinstance(new_objective_function, ObjectiveFunction)


