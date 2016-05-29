#Sean Billings, 2015


from backend import constraints
from backend.experiment import W1Experiment
from backend.objectiveFunctions import WeightedSumObjectiveFunction, IdealDifferentialObjectiveFunction
from backend.spea_optimizer import SpeaOptimizer
from backend.paretoFunctions import ParetoMaxFunction
from backend.photonicCrystalDesign import PhCWDesign


# absolute path to the mpb executable
mpb = "/Users/sean/documents/mpb-1.5/mpb/mpb"

# absolute path to the input ctl
inputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/W1_2D_v04.ctl.txt"

# absolute path to the output ctl
outputFile = "/Users/sean/UniversityOfOttawa/Photonics/PCWO/optimizerTestFile.txt"


# we define a general experiment object
# that we reuse whenever we need to make a command-line mpb call
# see experiment.py for functionality
experiment = W1Experiment(mpb, inputFile, outputFile)
# ex.setParams(paramVector)
experiment.setCalculationType('4') # accepts an int from 0 to 5
experiment.setBand(23)


# see constraints.py
constraintFunctions = [constraints.latticeConstraintsLD]

# Seed solutions
solutions = [{'r0': 0.214783, 'r1': 0.209531, 'r2': 0.201333, 'r3': 0.295405, 's3': 0.007753, 's2': 0.003418, 's1': 0.036686},
{'r0': 0.203795, 'r1': 0.2, 'r2': 0.2, 'r3': 0.269037, 's3': -0.006079, 's2': -0.00348, 's1': -0.070115},
{'r0': 0.203795, 'r1': 0.2, 'r2': 0.243051, 'r3': 0.269037, 's3': -0.006079, 's2': -0.00348, 's1': -0.070115},
{'r0': 0.2, 'r1': 0.2, 'r2': 0.206285, 'r3': 0.261162, 's3': -0.004591, 's2': -0.000385, 's1': -0.045711},
{'r0': 0.24906, 'r1': 0.2, 'r2': 0.205319, 'r3': 0.249118, 's3': -0.004591, 's2': -0.000385, 's1': -0.035766},
{'r0': 0.239884, 'r1': 0.2, 'r2': 0.275611, 'r3': 0.275994, 's3': -0.002681, 's2': -0.00337, 's1': -0.098542},
{'r0': 0.2, 'r1': 0.2, 'r2': 0.277873, 'r3': 0.280233, 's3': 0.00195, 's2': 0.001345, 's1': -0.032942},
{'r0': 0.24906, 'r1': 0.2, 'r2': 0.306868, 'r3': 0.249118, 's3': 0.000862, 's2': -0.001336, 's1': -0.048956},
{'r0': 0.237235, 'r1': 0.2, 'r2': 0.267718, 'r3': 0.261162, 's3': -0.006669, 's2': 0.004421, 's1': -0.064895},
{'r0': 0.211601, 'r1': 0.2, 'r2': 0.245645, 'r3': 0.249118, 's3': 0.00195, 's2': 0.001345, 's1': -0.046039},
{'r0': 0.203932, 'r1': 0.2, 'r2': 0.2, 'r3': 0.261162, 's3': -0.005686, 's2': -0.000619, 's1': -0.049269},
{'r0': 0.2, 'r1': 0.2, 'r2': 0.270556, 'r3': 0.256488, 's3': -0.006669, 's2': 0.003713, 's1': -0.064895},
{'r0': 0.221288, 'r1': 0.2, 'r2': 0.223155, 'r3': 0.296759, 's3': -0.002383, 's2': 0.004997, 's1': -0.029875},
{'r0': 0.233377, 'r1': 0.2, 'r2': 0.243644, 'r3': 0.261162, 's3': -0.004684, 's2': 0.004804, 's1': -0.045711},
{'r0': 0.2, 'r1': 0.2, 'r2': 0.270556, 'r3': 0.276756, 's3': -0.004591, 's2': -0.000385, 's1': -0.045909},
{'r0': 0.224803, 'r1': 0.2, 'r2': 0.2, 'r3': 0.261162, 's3': -0.006669, 's2': 0.003713, 's1': -0.045961},
{'r0': 0.211601, 'r1': 0.2, 'r2': 0.308447, 'r3': 0.249118, 's3': 0.000862, 's2': -0.002136, 's1': -0.046039},
{'r0': 0.234865, 'r1': 0.2, 'r2': 0.2, 'r3': 0.251053, 's3': -0.004591, 's2': -0.00416, 's1': -0.045909},
{'r0': 0.273167, 'r1': 0.2, 'r2': 0.2, 'r3': 0.249118, 's3': -0.003447, 's2': 0.004997, 's1': 0.040613},
{'r0': 0.224682, 'r1': 0.2, 'r2': 0.243644, 'r3': 0.261162, 's3': -0.003447, 's2': 0.006156, 's1': -0.084962},
{'r0': 0.2, 'r1': 0.2, 'r2': 0.283094, 'r3': 0.261162, 's3': -0.006669, 's2': 0.004421, 's1': -0.064895}]



population = []
for vector in solutions:
    pcw = PhCWDesign(vector, 0, constraintFunctions)
    population.append(pcw.copy_phc)

max_generation = 1 # number of iterations for SPEA
#population_size = 10 # number of solutions to consider in SPEA
pareto_archive_size = 40 # number of solutions to store in the SPEA PAS
tournament_selection_rate  = 5 # number of solutions to consider for evolution in SPEA

#Initialize objective function
#objFunc = IdealDifferentialObjectiveFunction(weights, experiment, ideal)
key_map = {}
key_map["ng0"] = "max"
key_map["delay"] = "max"
key_map["bandwidth"] = "max"
#key_map["loss_at_ng0"] = "min"

pareto_function = ParetoMaxFunction(experiment, key_map)

# Differential Evolution section

print "Starting SPEA"


optimizer = SpeaOptimizer(pareto_function)

optimizer.optimize(population,max_generation,tournament_selection_rate, pareto_archive_size)
