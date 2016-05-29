__author__ = 'sean'

import constraints
import waveGuideMPBOptimizer

solution = {'p2': 0.5, 'p3': 0.5, 'p1': 0.5, 'r0': 0.5, 'r1': 0.5, 'r2': 0.5, 'r3': 0.5, 's3': 0.5, 's2': 0.5, 's1': 0.5}

constraintFunctions = [constraints.latticeConstraintsLD]

print "Testing constraints"

waveGuideMPBOptimizer.createPopulation(constraintFunctions, 10000, solution)

print "Test Passed"
