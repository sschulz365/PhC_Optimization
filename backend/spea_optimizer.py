# Sean Billings, 2015

import random
import math
from optimizer import Optimizer
from abc import ABCMeta, abstractmethod
from photonicCrystalDesign import PhCWDesign
from paretoFunctions import ParetoFunction

class SpeaOptimizer(Optimizer):
    __metaclass__ = ABCMeta



    def __init__(self, pareto_function):
        self.pareto_function = pareto_function
        self.solutions = []


    def optimize(self, population, max_generation, tournament_selection_rate, pareto_archive_size):
        solutions = self.strength_pareto_evolutionary_algorithm(population, max_generation,
                                            tournament_selection_rate, pareto_archive_size)
        return solutions


    # evaluate the solutions in the population using SPEA
    # SPEA is an evolutionary algorithm that generates non-dominated (pareto) solutions in each generation
    # the pareto front is balanced by clustering of the objectives, I.E [2,1,0] is clustered with [2.1,1.1,0] in the pareto front
    # the number and quality of other solutions which a particular solutions dominates is summarized in the notion of the 'strength' of that solution
    # this measure of 'strength' is used to rank non-dominated solutions, and adds additional structure to the pareto front
    #
    # tournament_selection_rate can be tuned in order to affect the probability of a non-dominated solution being selected for evolution (crossover/mutation)
    # a higher tournament_selection_rate increases the probabiity that a non-dominated solution will be used for evolution,
    # but it also increases the probability that the same solution will be chosen twice
    def strength_pareto_evolutionary_algorithm(self, population, max_generation,
                                            tournament_selection_rate, pareto_archive_size):

        # pareto_set is the set of non-dominated solutions that are generated across generations
        pareto_set = []

        # iterate the SPEA evaluation routine over max_generation generations
        for i in range(0, max_generation):

            # evaluate population and pareto_set

            print "\n\nGeneration " + str(i)

            evaluated_population = []

            # evaluates the objectives of each solution in population and stores evaluated solutions in the form [objective_function_map, solution]
            # because dominance works in terms of maximization, the inverted loss 1/a is used
            for pcw in population:
                #objective_function_map = {}
                #print "DEBUG: Solution: " + str(pcw.solution_vector)
                self.pareto_function.evaluate(pcw)


                objective_function_results = pcw.figures_of_merit

                # objective_function_results is in the form
                # [bandwidthNormalized, ng0, avgLoss, bandWidthRatio, loss_at_ng0]      (Aug 11, 2015)

                #objective_function_map["bandwidth"] = objective_function_results[0]
                #objective_function_map["group_index"] = objective_function_results[1]
                #objective_function_map["average_loss*"] = 1/objective_function_results[2]
                #objective_function_map["BGP"] = objective_function_results[3]
                #objective_function_map["loss_at_group_index"] = 1/objective_function_results[4]
                #objective_function_map["delay"] = objective_function_results[5]
                #evaluated_population.append([objective_function_map, solution])

                evaluated_population.append(pcw)


                # display solution and results in the console
                print "Solution: " + str(pcw.solution_vector) + "\n"


                print "Objectives: "
                for key in self.pareto_function.key_list.keys():
                    if self.pareto_function.key_list[key] == "min":
                        if pcw.figures_of_merit[key] == 0:
                            print str(key) +": undefined"
                        else:
                            print str(key) +": " + str(float("{0:.4f}".format(float(1)/pcw.figures_of_merit[key])))
                    else:
                        print str(key) +": " + str(float("{0:.4f}".format(pcw.figures_of_merit[key])))
                print "\n"
            # end population evaluation


            # determine non-dominated solutions and store them in pareto_set
            updated_pareto_set = self.extract_non_dominated_solutions(evaluated_population, pareto_set)

            # use clustering to reduce the size of the pareto set, if necessary
            if len(updated_pareto_set) > pareto_archive_size:
                updated_pareto_set = self.cluster_reduce(updated_pareto_set, pareto_archive_size)



            pareto_set = updated_pareto_set[:] # reference safety precaution

            # print the results in the pareto_set
            j= 1
            for pcw in pareto_set:
                #objective_function_map = solution[0]
                print "\nPareto Solution " + str(j) + ": "+ str(pcw.solution_vector)+ "\n"
                j += 1
                print "Non-Dominated Objectives: "
                for key in self.pareto_function.key_list.keys():
                    if self.pareto_function.key_list[key] == "min":
                        if pcw.figures_of_merit[key] == 0:
                            print str(key) +": undefined"
                        else:
                            print str(key) +": " + str(float("{0:.4f}".format(float(1)/pcw.figures_of_merit[key])))
                    else:
                        print str(key) +": " + str(float("{0:.4f}".format(pcw.figures_of_merit[key])))
                print "\n"

            # generate the new population using binary tournament selection
            if i < max_generation:
                # evaluate the 'strength' of solution in the population and pareto set
                # in order determine which solutions will be selected for evolution
                scored_pareto_set, scored_population = self.evaluate_fitness(updated_pareto_set, evaluated_population)

                population = self.evolve(tournament_selection_rate, scored_pareto_set, scored_population, len(evaluated_population))


        # executes after max_generation iterations of the SPEA routine, have executed

        return pareto_set



    # determine which solutions are non-dominated in { population U pareto_set }
    # attempts to extract only unique solutions
    # a solution is non-dominated if there is no other solution which is "as good or better" in every objective
    def extract_non_dominated_solutions(self, population, pareto_set):

        non_dominated_solutions = []
        non_dominated_population = []

        # determine the unique non-dominated solutions in {population}
        # store results in non_dominated_population
        for pcw in population:
            include = True
            for test_pcw in population:
                if test_pcw != pcw:
                    # print test_solution
                    if self.pareto_function.dominates(test_pcw, pcw):
                        include = False

            if include:
                non_dominated_population.append(pcw)

        # determine which solutions in non_dominated_population are also non-dominated in { pareto_set}
        # store results in non_dominated_solutions
        for pcw in non_dominated_population:
            include = True
            for test_pcw in pareto_set:
                if test_pcw != pcw:
                    # print test_solution
                    if self.pareto_function.dominates(test_pcw, pcw):
                        include = False
            if include:
                for already_solution in non_dominated_solutions:
                    if pcw == already_solution:
                        include = False
                if include:
                    non_dominated_solutions.append(pcw)

        # determine which unique solutions in the pareto_set are non-dominated in { pareto_set U non_dominated_population }
        # stores results in non_dominated_solutions
        for solution in pareto_set:
            include = True
            for test_solution in non_dominated_population:
                if test_solution != solution:
                    # print test_solution
                    if self.pareto_function.dominates(test_solution, solution):
                        include = False
            if include:
                for already_solution in non_dominated_solutions:
                    if solution == already_solution:
                        include = False
                if include:
                    non_dominated_solutions.append(solution)


        return non_dominated_solutions


    # reduce the size of the pareto set via clustering until the number of clusters is pareto_archive_size
    # then select centroids from each cluster to create a new pareto set, returned as reduced_pareto_set
    def cluster_reduce(self, updated_pareto_set, pareto_archive_size):

        # add all solutions to the indexed cluster_set
        cluster_set = {}
        # print "\n" # + str(updated_pareto_set) + "\n" # sanity check
        for i in range (0, len(updated_pareto_set)):
            cluster_set[i] = [updated_pareto_set[i]]
        #print cluster_set.keys() # sanity check
        #print "\n" # sanity check
        cluster_size = len(cluster_set.keys())

        # reduce the number of clusters to pareto_archive_size
        while cluster_size > pareto_archive_size:
            min_distance = 1000000

            keys = cluster_set.keys()
            for i in keys:
                for j in keys:
                    if i != j:
                        cluster_dist = self.cluster_distance(cluster_set[i], cluster_set[j])
                        if cluster_dist < min_distance:
                            cluster_a_index = i
                            cluster_b_index = j
                            min_distance = cluster_dist
            new_cluster = []
            for a in cluster_set[cluster_a_index]:
                new_cluster.append(a)

            for b in cluster_set[cluster_b_index]:
                new_cluster.append(b)

            cluster_set.pop(cluster_b_index) # potentially wrong
            cluster_set[cluster_a_index] = new_cluster
            #print cluster_set.keys() # sanity check
            #print len(cluster_set.keys()) # should be updated from above
            cluster_size = len(cluster_set.keys())
        # end of cluster generation

        # select a centroid from each cluster and store them in reduced_pareto_set

        reduced_pareto_set = []

        for key in cluster_set.keys():
            #TODO: do better
            i = random.randint(0,(len(cluster_set[key])-1)) # could also select most central points
            reduced_pareto_set.append(cluster_set[key][i])

        return reduced_pareto_set


    # use the notion of 'strength' in order to rank and score the non-dominated solutions
    # for use in the latter binary tournament selection for evolution
    def evaluate_fitness(self, updated_pareto_set, evaluated_population):

        scored_pareto_set = []
        scored_population = []
        for solution in updated_pareto_set:
            count = 0
            for pop in evaluated_population:
                if self.pareto_function.strongly_dominates(solution, pop):
                    count = count + 1
            scored_pareto_set.append([count, solution])

        for pop in evaluated_population:
            for scored_solution in scored_pareto_set:
                sum = 0
                if self.pareto_function.strongly_dominates(pop, scored_solution[1]):
                    sum = sum + scored_solution[0]
                scored_population.append([sum, pop])

        return scored_pareto_set, scored_population


    # use mutation and crossover to develop a population for the next generation
    def evolve(self, tournament_selection_rate, scored_pareto_set, scored_population, population_size):
        updated_population = []
        # print scored_population
        mating_pool = scored_population[:]
        mating_pool.extend(scored_pareto_set[:])
        while len(updated_population) < population_size:
            selected_solutions = []
            for i in range(0, tournament_selection_rate):
                a_selection = mating_pool[random.randint(0,(len(mating_pool)-1))]
                selected_solutions.append(a_selection)
            # imagine probability of crossover as 0.2%
            rand = random.random()
            if rand < 0.3:
                # crossover

                # find top 2 scoring solutions from selected_solutions
                parent_a_score = 0
                parent_b_score = 0
                parent_a = selected_solutions[0]
                for i in range(1, len(selected_solutions)):
                    solution = selected_solutions[i]
                    if solution[0] > parent_a_score:
                        parent_b = parent_a
                        parent_b_score = parent_a_score
                        parent_a = solution
                        parent_a_score = solution[0]

                    elif solution[0] >= parent_b_score:
                        parent_b = solution
                        parent_b_score = solution[0]
                # fix

                # parent_a = mating_pool[random.randint(0,len(mating_pool))]
                # parent_b = mating_pool[random.randint(0,len(mating_pool))]
                updated_population.append(self.crossover(parent_a[1], parent_b[1]))

            else:
                # mutate
                # parent_a = mating_pool[random.randint(0,len(mating_pool))]

                # find top scoring solution from selected_solutions
                parent_a_score = 0
                parent_a = selected_solutions[0]
                for i in range(1, len(selected_solutions)):
                    solution = selected_solutions[i]
                    if solution[0] > parent_a_score:
                        parent_a = solution
                        parent_a_score = solution[0]
                #
                updated_population.append(self.mutate(parent_a[1]))

        return updated_population


    # a and b are in the form [ obj_function_results, parameter_map]
    # obj_function_results is the results from parseObjParams in mpbParser.py
    # parameter_map is defined in the runOptimizer.py module
    # both are dictionaries
    # this method simply evaluates whether a dominates b
    def dominates(self, a, b):
        #print "\n" + str(a)
        for key in a.keys():
            if math.fabs(a[key]) < math.fabs(b[key]):
                return False

        return True


    # a and b are in the form [ obj_function_results, parameter_map]
    # obj_function_results is the results from parseObjParams in mpbParser.py
    # parameter_map is defined in the runOptimizer.py module
    # both are dictionaries
    # this method simply evaluates whether a strongly dombinates b
    def strongly_dominates(self, a, b):
        # print "\n" + str(a) + "dominates\n" + str(b) + "?\n"
        for key in a.keys():
            if math.fabs(a[key]) <= math.fabs(b[key]):
                return False

        return True


    # a and b are PhCW
    # obj_function_results is the results from parseObjParams in mpbParser.py
    # parameter_map is defined in the runOptimizer.py module
    # both are dictionaries
    # this method simply computes the L2 (Euclidian) norm between the vectors of objectives for a and b
    # and returns the result as distance
    def l2_norm(self, a, b):
        distance = 0
        for key in a.figures_of_merit.keys():
            distance += (float("{0:.4f}".format(math.fabs(a.figures_of_merit[key]))) - float("{0:.4f}".format(math.fabs(b.figures_of_merit[key]))))**2
        distance = math.sqrt(distance)
        return distance


    # parent_a, parent_b are in the form [ obj_function_results, parameter_map]
    # this method performs a randomized crossover of parent_a and parent_b
    # the result is returned as new_solution
    def crossover(self, parent_a, parent_b):
        new_solution = {}

        # print parent_a[1]
        for key in parent_a.solution_vector.keys():
            rand = random.random()
            if rand > 0.5:
                new_solution[key] = parent_a.solution_vector[key]
            else:
                new_solution[key] = parent_b.solution_vector[key]

        new_pcw = PhCWDesign(new_solution,0,parent_a.constraints)
        new_pcw.constrain()
        return new_pcw


    # parent_a is a pcw
    # this method performs a randomized mutation of parent_a
    # the result is returned as new_pcw
    def mutate(self, parent_a):
        new_solution = {}
        for key in parent_a.solution_vector.keys():
            rand = random.random()
            if rand > 0.5:
                # mutate
                polarity = 1
                if 'r' not in key:
                    polarity_rand = random.random() - 0.2

                    if polarity_rand < 0:
                        polarity = -1

                rand2 = random.random()
                rand2 = polarity*(math.sqrt(rand2 + 0.5))

                new_solution[key] = float("{0:.6f}".format(rand2*(parent_a.solution_vector[key] + 0.001)))

            else:
                new_solution[key] = parent_a.solution_vector[key]

        new_pcw = PhCWDesign(new_solution,0,parent_a.constraints)
        new_pcw.constrain()

        return new_pcw


    # cluster_a and cluster_B are arrays of solutions
    # each solution is of the form [obj_function_results, parameter_map]
    # This method simply computes cluster distance using the minimum distance between
    # any two vectors (of objectives for a solution), where one vector is from each cluster
    # other metrics like average distance could be used in place of minimum distance
    def cluster_distance(self, cluster_a, cluster_b):

        min_distance = 100000
        for pcw_a in cluster_a:
            for pcw_b in cluster_b:
                distance = self.l2_norm(pcw_a, pcw_b)
                if distance < min_distance:
                    min_distance = distance

        return min_distance

