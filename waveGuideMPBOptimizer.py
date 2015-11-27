#Sean Billings, 2015

# This module defines a set of optimization routines that
# can be used to optimize a given waveguide as defined by
# a ctl file designed for use in the MIT Photonics Band
# software library.

import random
import numpy
import constraints
import copy
import mpbParser
import math


# creates a population which is a dictionary that maps parameters to generated random values
# see paramMap in runOptimizer.py for a population sample example
def createPopulation(constraintFunctions, population_size, vector_archetype):
    population = [vector_archetype.copy() for i in range (0, population_size)]
    for i in range(1, population_size):
        new_vector =  population[i]
        #boundary = 1
        for j in new_vector.keys():
            if 'r' in j:

                new_vector[j] = (math.log(1-random.random())/-2)/5 + 0.2
            else:
                scale = random.random() - 0.5
                new_vector[j] = (math.log(1-random.random())/-10)*scale
            # exponetial distribution with lamda = 10
            
        population[i] = constraints.fix(new_vector.copy(), constraintFunctions) # from constraints


    return population


# Differential Evolution Optimizer
# Creates and returns a set of solutions which minimize the passed objectiveFunction
# The results are not guaranteed local or global minima, but the DE algorithm
# provides good coverage of the solution space, and can provide great results

def differentialEvolution(constraintFunctions, objectiveFunction,
                          max_generation, population_size, 
                          random_update, vector_archetype, 
                          elite_size, experiment):

    # generation 0
    # initialize population of numpy arrays
    print "Initializing Population..."
    population = createPopulation(constraintFunctions,
                                  population_size,
                                  vector_archetype)
    elites = {} # hashmap of scores and vectors
    top_scores = []
    #score population
    population_score = {} # hashmap of population indexes with scores

    for i in range(0, population_size):
        print "ID: " + str(i)
        print population[i]
        population_score[i] = objectiveFunction.evaluate(population[i])
        results = population_score[i]
        solution_score = results[0]
        bandwidth = results[1]
        group_index = results[2]
        avgLoss = results[3] # average loss
        bandwidth_group_index_product = results[4] #BGP
        loss_at_ng0 = results[5] # loss at group index
        print "\nScore: " + str(solution_score)
        print "\nNormalized Bandwidth: " +  str(bandwidth)
        print "\nGroup Index: " + str(group_index)
        print "\nAverage Loss: " + str(avgLoss)
        print "\nLoss at Group Index: " + str(loss_at_ng0)
        print "\nBGP: " + str(bandwidth_group_index_product)
        print "\n"
        # each step will require a simulatation in MPB
        # print "Score: " + str(population_score[i])
        
        # supplement elites
        if i < elite_size:
            elites[population_score[i][0]] = population[i]
            top_scores.append(population_score[i])
            
    # iteratively update the population for each generation
    
    for generation in range(1, max_generation + 1):
        print "\nGeneration: " + str(generation)

        # update vectors (performs a standard DE mutation operator)
        print "\nUpdating Population..."
        for j in range(0, population_size):
            
            # generate random vectors (indices) xi from elites
            x1 = numpy.random.randint(1, elite_size)
            x2 = numpy.random.randint(1, elite_size)
            x3 = numpy.random.randint(1, elite_size)
            # could base on scoring metric


            # could clean this up to be more efficient
            while x1 == x2 or x2 == x3 or x1 == x3 or j == x1 or j == x2 or j == x3:
                x1 = numpy.random.randint(1, elite_size)
                x2 = numpy.random.randint(1, elite_size)
                x3 = numpy.random.randint(1, elite_size)

            # Select vector from population
            u = population[j].copy()
            
            # Update entries in vector via differential                  
            for i in vector_archetype.keys():               
                if random.random() < random_update:
                    # scaling method
                    # u[i] = math.sqrt(((population[x3])[i])*(population[x1])[i])

                    # vector addition method
                    u[i] = (population[x3])[i] + ((population[x1])[i] - (population[x2])[i])

            # fix is from the constraints library
            # it will map parameters in u back to acceptable bounds
            u = constraints.fix(u, constraintFunctions)
            u_score = objectiveFunction.evaluate(u)

            # determine whether to update the vector at j with the new vector u
            if population_score[j][0] > u_score[0]:
                population[j] = u.copy()
                population_score[j] = u_score
            
        #end for j

        print "\nGenerating Elites..."
        # There is a bug that involves not being able to fill the elites bag
        # to maximum capacity in the elite update/generate code below
        
        # compute the worst current elite score (max_score)
        max_score = 0
        for scores in top_scores:
            if scores[0] > max_score:
                max_score = scores[0]

        # update elites with best new solutions from the population
        for k in range(0, population_size):
            if population_score[k][0] <= max_score:
                elites[population_score[k][0]] = population[k].copy()
                top_scores.append(population_score[k])
            
        ## generate new elites

        # tricky set conversion to create unique, sorted version of top_scores
        unique_top_scores = list()
        top_scores = sorted(top_scores, key=lambda score: score[0])
        map(lambda x: not x in unique_top_scores and unique_top_scores.append(x), top_scores)

        top_scores = unique_top_scores

        top_score_indexes = numpy.unique(x[0] for x in top_scores)  #numpy.unique returns the sorted unique elements from top_scores
        print "\nTop Scores: " + str(top_scores)
       
        
        nextgen_elites = {}
        nextgen_top_scores = []
                        
        for i in range(0, elite_size):
            nextgen_elites[top_scores[i][0]] = elites[top_scores[i][0]]
            nextgen_top_scores.append(top_scores[i])
        elites = nextgen_elites.copy()
        top_scores = copy.deepcopy(nextgen_top_scores)
        
        print "\nElites: "
        for i in range(0, len(top_scores)):
            print "\nElite: " + str(elites[top_scores[i][0]])
            results = top_scores[i]
            solution_score = results[0]
            bandwidth = results[1]
            group_index = results[2]
            avgLoss = results[3] # average loss
            bandwidth_group_index_product = results[4] #BGP
            loss_at_ng0 = results[5] # loss at group index
            print "\nScore: " + str(solution_score)
            print "\nNormalized Bandwidth: " +  str(bandwidth)
            print "\nGroup Index: " + str(group_index)
            print "\nAverage Loss: " + str(avgLoss)
            print "\nLoss at Group Index: " + str(loss_at_ng0)
            print "\nBGP: " + str(bandwidth_group_index_product)
            # end for  generation
                                                   
    #final_elites = []
    #for i in range(0, elite_size):
    #    final_elites.append(elites[top_scores[i]])
                                                   
    return elites.values()


# minimizes the score a set of vectors using the designated objectiveFunction
# according to a set of stopping conditions
# and returns the optimized vectors as a set of solutions
            
def gradientDescentAlgorithm(objectiveFunction, constraintFunctions,
                             population, descent_scaler,
                             completion_scaler, alpha_scaler):

    max_iterations = 5
    solutions = []
    j = 1
    # optimize vectors with gradient descent
    for vector in population:
        print "\nPerforming Gradient Descent on " + str(vector)
        print "\nVector " + str(j) + " of " + str(len(population))
        j += 1
        # print vector
        vector_score = objectiveFunction.evaluate(vector)[0]
        new_vector_score = vector_score
        print "\nInitial score: " + str(vector_score)
        # could keep track of scores
        i = 1       
        while i <= max_iterations:
            print "\n\nIteration: " + str(i)
            # the Gradient Descent method recursively improves 'vector' until a stopping condition is met
            # (see above for method specifications)
            result = gradientDescent(objectiveFunction,
                                     constraintFunctions,
                                     vector,
                                     new_vector_score,
                                     descent_scaler,
                                     completion_scaler,
                                     alpha_scaler)
            checkVector = vector.copy()
            vector = result[0]
            new_vector_score = result[1]
            if vector == checkVector:
                i = 6
                # This results means that the solution is approximately convergent to a local minima
                print "Convergent iteration results"
                print vector
                print "\nFinal Score: " + str(new_vector_score)
                print "Total Improvement: " + str(new_vector_score - vector_score)
     
            else:               
                print "\nIiteration " + str(i) + " results"
                print vector
                print "\nScore: " + str(new_vector_score)
                print "Total Improvement: " + str(new_vector_score - vector_score)
            i += 1

        # store our vector in a set of solutions            
        solutions.append(vector)
        
    # end for vector
    return solutions


# Performs a version of the (steepest) gradient descent
# with an augmented backtracking algorithm structure for a safe and efficient
# gradient descent operation on 'vector'
# returns a new vector which can be fed into a general gradient descent algorithm
# This method is time consuming but the convergence is nice and it can approximate
# local minima in the neighbourhood of the passed vector


def gradientDescent(objectiveFunction, constraintFunctions,
                    vector, initial_score, descent_scaler,
                    completion_scaler, alpha_scaler):

    #establish the base score to improve upon
    current_score = initial_score

    if current_score > 10000: # arbitrary max score
        print "\nDescent not achievable"
        return [vector, current_score]

    # create next_vector to be used in gradient descent
    next_vector = vector.copy()

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
        # constraints.fix(vectorPlusDeltaKey, constraintFunctions)
        deltaPlusScore = (objectiveFunction.evaluate(vectorPlusDeltaKey)[0])

        # attempt to deal with parsing errors by recalculating deltaPlusScore
        if deltaPlusScore > 10000:
                   deltaPlusScore = (objectiveFunction.evaluate(vectorPlusDeltaKey)[0])


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
        next_vector[key] = max([0, (vector[key] - gradientValues[key])])
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

    constraints.fix(next_vector,constraintFunctions)
    next_score = objectiveFunction.evaluate(next_vector)[0]

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
           next_vector[key] = max( [ 0, (vector[key] - alpha_scaler*gradientValues[key])])

        constraints.fix(next_vector, constraintFunctions)

        next_score = objectiveFunction.evaluate(next_vector)[0]

        gradient_factor = completion_scaler*alpha_scaler*gradient_innerProduct
        print "\n" + str(alpha_scaler)
        print next_vector
        alpha_scaler = descent_scaler*alpha_scaler
        attempts+= 1
    
    if attempts > 4:
        print "\nDescent not achievable"
        return [vector, current_score]
    
    # if a vectors score already satisfies the wolfe conditions,
    # then we scale up the value of the gradient descent
    if attempts == 0:
        # scale  the gradient at the values 10, 100, 1000, 10000
        # where 10000 * gradientValues[key] is the actual gradient value
        # in contrast to the reduced (scaled by 0.0001) gradient that is computed
        attempts = 1
        ascent_scaler = 10
        ascent_vector = next_vector.copy()

        for key in vector.keys():
            ascent_vector[key] = max( [ 0, (vector[key] - (ascent_scaler)*gradientValues[key])])

        constraints.fix(ascent_vector, constraintFunctions)

        ascent_score = objectiveFunction.evaluate(ascent_vector)[0]
        gradient_factor = completion_scaler*ascent_scaler*gradient_innerProduct
        
        print "Maximizing Gradient"
        print "\n" + str(ascent_scaler)
        print ascent_vector
        ascent_scaler = 10*ascent_scaler

        while ascent_score > (current_score + gradient_factor) and attempts < 5:
            ascent_vector = next_vector.copy()
            for key in vector.keys():
                ascent_vector[key] = max( [ 0, (vector[key] - (ascent_scaler)*gradientValues[key])])
                ascent_vector[key] = min( [ 1, (vector[key] - (ascent_scaler)*gradientValues[key])])
            constraints.fix(ascent_vector, constraintFunctions)

            ascent_score = objectiveFunction.evaluate(ascent_vector)[0]

            gradient_factor = completion_scaler*ascent_scaler*gradient_innerProduct
            print "Scaling Ascent"
            print "\n" + str(ascent_scaler)
            print ascent_vector
            ascent_scaler = 10*ascent_scaler
            attempts+= 1
            
        if ascent_score < next_score:
            return [ascent_vector, ascent_score]
        

    
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

    return [next_vector, next_score]


# evaluate the solutions in the population using SPEA
# SPEA is an evolutionary algorithm that generates non-dominated (pareto) solutions in each generation
# the pareto front is balanced by clustering of the objectives, I.E [2,1,0] is clustered with [2.1,1.1,0] in the pareto front
# the number and quality of other solutions which a particular solutions dominates is summarized in the notion of the 'strength' of that solution
# this measure of 'strength' is used to rank non-dominated solutions, and adds additional structure to the pareto front
#
# tournament_selection_rate can be tuned in order to affect the probability of a non-dominated solution being selected for evolution (crossover/mutation)
# a higher tournament_selection_rate increases the probabiity that a non-dominated solution will be used for evolution,
# but it also increases the probability that the same solution will be chosen twice
def strength_pareto_evolutionary_algorithm(population, experiment,
                                           constraintFunctions, max_generation,
                                           pareto_archive_size, tournament_selection_rate):


    # pareto_set is the set of non-dominated solutions that are generated across generations
    pareto_set = []

    # iterate the SPEA evaluation routine over max_generation generations
    for i in range(0, max_generation):

        # evaluate population and pareto_set

        print "\n\nGeneration " + str(i)

        evaluated_population = []

        # evaluates the objectives of each solution in population and stores evaluated solutions in the form [objective_function_map, solution]
        # because dominance works in terms of maximization, the inverted loss 1/a is used
        for solution in population:
            objective_function_map = {}

            experiment.setParams(solution)
            experiment.perform()
            objective_function_results = mpbParser.parseObjFunctionParams(experiment)

            # objective_function_results is in the form
            # [bandwidthNormalized, ng0, avgLoss, bandWidthRatio, loss_at_ng0]      (Aug 11, 2015)

            #objective_function_map["bandwidth"] = objective_function_results[0]
            objective_function_map["group_index"] = objective_function_results[1]
            #objective_function_map["average_loss*"] = 1/objective_function_results[2]
            #objective_function_map["BGP"] = objective_function_results[3]
            objective_function_map["loss_at_group_index"] = 1/objective_function_results[4]
            #objective_function_map["delay"] = objective_function_results[5]
            evaluated_population.append([objective_function_map, solution])


            # display solution and results in the console
            print "\nSolution: " + str(solution) + "\n"


            print "Objectives: "
            for key in objective_function_map:
                if key == "loss_at_group_index" or key == "average_loss":
                    if objective_function_map[key] == 0:
                        objective_function_map[key] = 0.000000001
                    print str(key) +": " + str(float("{0:.4f}".format(float(1)/objective_function_map[key])))
                else:
                    print str(key) +": " + str(float("{0:.4f}".format(objective_function_map[key])))
            print "\n"
        # end population evaluation


        # determine non-dominated solutions and store them in pareto_set
        updated_pareto_set = extract_non_dominated_solutions(evaluated_population, pareto_set)

        # use clustering to reduce the size of the pareto set, if necessary
        if len(updated_pareto_set) > pareto_archive_size:
            updated_pareto_set = cluster_reduce(updated_pareto_set, pareto_archive_size)



        pareto_set = updated_pareto_set[:] # reference safety precaution

        # print the results in the pareto_set
        for solution in pareto_set:
            objective_function_map = solution[0]
            print "\nPareto Solution: " + str(solution[1]) + "\n"
            print "Non-Dominated Objectives: "
            for key in objective_function_map:
                if key == "loss_at_group_index" or key == "average_loss":
                    print str(key) +": " + str(float("{0:.4f}".format(float(1)/objective_function_map[key])))
                else:
                    print str(key) +": " + str(float("{0:.4f}".format(objective_function_map[key])))


        # generate the new population using binary tournament selection
        if i < max_generation:
            # evaluate the 'strength' of solution in the population and pareto set
            # in order determine which solutions will be selected for evolution
            scored_pareto_set, scored_population = evaluate_fitness(updated_pareto_set, evaluated_population)

            population = evolve(tournament_selection_rate, scored_pareto_set, scored_population, len(evaluated_population), constraintFunctions)


    # executes after max_generation iterations of the SPEA routine, have executed

    """
    # compute the final set of non_dominated_solutions

    spea_solutions = extract_non_dominated_solutions(evaluated_population, pareto_set)

    for solution in spea_solutions:

        if "average_loss" in solution[0].keys():
            solution[0]["average_loss"] = 1/solution[0]["average_loss"]
        if "loss_at_group_index" in solution[0].keys():
            solution[0]["loss_at_group_index"] = 1/solution[0]["loss_at_group_index"]
    """

    return [ x[1] for x in pareto_set]



# determine which solutions are non-dominated in { population U pareto_set }
# attempts to extract only unique solutions
# a solution is non-dominated if there is no other solution which is "as good or better" in every objective
def extract_non_dominated_solutions(population, pareto_set):

    non_dominated_solutions = []
    non_dominated_population = []

    # determine the unique non-dominated solutions in {population}
    # store results in non_dominated_population
    for solution in population:
        include = True
        for test_solution in population:
            if test_solution[1] != solution[1]:
                # print test_solution
                if dominates(test_solution[0], solution[0]):
                    include = False

        if include:
            non_dominated_population.append(solution)

    # determine which solutions in non_dominated_population are also non-dominated in { pareto_set}
    # store results in non_dominated_solutions
    for solution in non_dominated_population:
        include = True
        for test_solution in pareto_set:
            if test_solution[1] != solution[1]:
                # print test_solution
                if dominates(test_solution[0], solution[0]):
                    include = False
        if include:
            for already_solution in non_dominated_solutions:
                if solution[1] == already_solution[1]:
                    include = False
            if include:
                non_dominated_solutions.append(solution)

    # determine which unique solutions in the pareto_set are non-dominated in { pareto_set U non_dominated_population }
    # stores results in non_dominated_solutions
    for solution in pareto_set:
        include = True
        for test_solution in non_dominated_population:
            if test_solution[1] != solution[1]:
                # print test_solution
                if dominates(test_solution[0], solution[0]):
                    include = False
        if include:
            for already_solution in non_dominated_solutions:
                if solution[1] == already_solution[1]:
                    include = False
            if include:
                non_dominated_solutions.append(solution)


    return non_dominated_solutions


# reduce the size of the pareto set via clustering until the number of clusters is pareto_archive_size
# then select centroids from each cluster to create a new pareto set, returned as reduced_pareto_set
def cluster_reduce(updated_pareto_set, pareto_archive_size):

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
                    cluster_dist = cluster_distance(cluster_set[i], cluster_set[j])
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
        i = random.randint(0,(len(cluster_set[key])-1)) # could also select most central points
        reduced_pareto_set.append(cluster_set[key][i])

    return reduced_pareto_set


# use the notion of 'strength' in order to rank and score the non-dominated solutions
# for use in the latter binary tournament selection for evolution
def evaluate_fitness(updated_pareto_set, population):

    scored_pareto_set = []
    scored_population = []
    for solution in updated_pareto_set:
        count = 0
        for pop in population:
            if strongly_dominates(solution[0], pop[0]):
                count = count + 1
        scored_pareto_set.append([count, solution])

    for pop in population:
        for scored_solution in scored_pareto_set:
            sum = 0
            if strongly_dominates(pop[0], scored_solution[1][0]):
                sum = sum + scored_solution[0]
            scored_population.append([sum, solution])

    return scored_pareto_set, scored_population


# use mutation and crossover to develop a population for the next generation
def evolve(tournament_selection_rate, scored_pareto_set, scored_population, population_size, constraintFunctions):
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
            updated_population.append(crossover(parent_a[1], parent_b[1], constraintFunctions))

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
            updated_population.append(mutate(parent_a, constraintFunctions))

    return updated_population


# a and b are in the form [ obj_function_results, parameter_map]
# obj_function_results is the results from parseObjParams in mpbParser.py
# parameter_map is defined in the runOptimizer.py module
# both are dictionaries
# this method simply evaluates whether a dominates b
def dominates(a, b):
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
def strongly_dominates(a, b):
    # print "\n" + str(a) + "dominates\n" + str(b) + "?\n"
    for key in a.keys():
        if math.fabs(a[key]) <= math.fabs(b[key]):
            return False

    return True


# a and b are in the form [ obj_function_results, parameter_map]
# obj_function_results is the results from parseObjParams in mpbParser.py
# parameter_map is defined in the runOptimizer.py module
# both are dictionaries
# this method simply computes the L2 (Euclidian) norm between the vectors of objectives for a and b
# and returns the result as distance
def l2_norm(a, b):
    distance = 0
    for key in a[0].keys():
        distance += (float("{0:.4f}".format(a[0][key])) - float("{0:.4f}".format(b[0][key])))**2
    distance = math.sqrt(distance)
    return distance


# parent_a, parent_b are in the form [ obj_function_results, parameter_map]
# this method performs a randomized crossover of parent_a and parent_b
# the result is returned as new_solution
def crossover(parent_a, parent_b, constraintFunctions):
    new_solution = {}

    # print parent_a[1]
    for key in parent_a[1].keys():
        rand = random.random()
        if rand > 0.5:
            new_solution[key] = parent_a[1][key]
        else:
            new_solution[key] = parent_b[1][key]
    constraints.fix(new_solution, constraintFunctions)
    return new_solution


# parent_a is in the form [ obj_function_results, parameter_map]
# this method performs a randomized mutation of parent_a
# the result is returned as new_solution
def mutate(parent_a, constraintFunctions):
    new_solution = {}
    for key in parent_a[1][1].keys():
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

            new_solution[key] = float("{0:.6f}".format(rand2*(parent_a[1][1][key] + 0.001)))

        else:
            new_solution[key] = parent_a[1][1][key]
    constraints.fix(new_solution, constraintFunctions)
    return new_solution


# cluster_a and cluster_B are arrays of solutions
# each solution is of the form [obj_function_results, parameter_map]
# This method simply computes cluster distance using the minimum distance between
# any two vectors (of objectives for a solution), where one vector is from each cluster
# other metrics like average distance could be used in place of minimum distance
def cluster_distance(cluster_a, cluster_b):

    min_distance = 100000
    for vector_a in cluster_a:
        for vector_b in cluster_b:
            distance = l2_norm(vector_a, vector_b)
            if distance < min_distance:
                min_distance = distance

    return min_distance

