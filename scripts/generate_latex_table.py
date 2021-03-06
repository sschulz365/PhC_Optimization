__author__ = 'sean'


from backend.paretoFunctions import ParetoMaxFunction
from backend.experiment import W1Experiment
from backend import constraints
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

vectors = [{'p2': 0.305676, 'p3': -0.001217, 'p1': 0.226142, 'r0': 0.20691600000000002, 'r1': 0.20001000000000002, 'r2': 0.277464, 'r3': 0.227303, 's3': 0.02538, 's2': 0.161438, 's1': 0.198196},
           {'p2': -0.042182, 'p3': 0.005852, 'p1': -0.135591, 'r0': 0.233088, 'r1': 0.200446, 'r2': 0.383809, 'r3': 0.200118, 's3': -0.065357, 's2': -0.07563, 's1': -0.069276},
           {'p2': 0.178789, 'p3': 0.030898, 'p1': 0.124789, 'r0': 0.22809, 'r1': 0.237711, 'r2': 0.4, 'r3': 0.200028, 's3': 0.076132, 's2': -0.097793, 's1': 0.063853},
           {'p2': -0.12578699999999998, 'p3': -0.050950999999999996, 'p1': 0.13126800000000002, 'r0': 0.20001000000000002, 'r1': 0.20001000000000002, 'r2': 0.280337, 'r3': 0.259042, 's3': -0.052989999999999995, 's2': -0.141012, 's1': -0.15109299999999998},
           {'p2': 0.005054, 'p3': 0.004457, 'p1': -0.135639, 'r0': 0.232882, 'r1': 0.20002, 'r2': 0.383711, 'r3': 0.2, 's3': -0.065592, 's2': -0.075703, 's1': -0.069698},
           {'p2': 0.211282, 'p3': -0.042654, 'p1': 0.124609, 'r0': 0.20001, 'r1': 0.2, 'r2': 0.310886, 'r3': 0.2, 's3': -0.078041, 's2': -0.095444, 's1': -0.13245},
           {'p2': 0.02888, 'p3': 0.028308, 'p1': -0.141661, 'r0': 0.237307, 'r1': 0.202992, 'r2': 0.38373, 'r3': 0.2, 's3': -0.046709, 's2': -0.10155, 's1': -0.062486},
           {'p2': 0.005019, 'p3': 0.006606, 'p1': -0.135408, 'r0': 0.23733200000000002, 'r1': 0.20001000000000002, 'r2': 0.383755, 'r3': 0.20001000000000002, 's3': -0.046683999999999996, 's2': -0.094848, 's1': -0.062460999999999996},
           {'p2': 0.16922700000000002, 'p3': 0.030898, 'p1': 0.124609, 'r0': 0.22791, 'r1': 0.23769300000000002, 'r2': 0.40001000000000003, 'r3': 0.20001000000000002, 's3': 0.094081, 's2': -0.097833, 's1': 0.063673},
           {'p2': 0.005025, 'p3': 0.006642, 'p1': -0.113161, 'r0': 0.237332, 'r1': 0.20001, 'r2': 0.383809, 'r3': 0.200064, 's3': -0.046648, 's2': -0.098852, 's1': -0.062419},
           {'p2': -0.183304, 'p3': -0.050531999999999994, 'p1': -0.14166099999999998, 'r0': 0.20001000000000002, 'r1': 0.260614, 'r2': 0.355705, 'r3': 0.20001000000000002, 's3': 0.08078099999999999, 's2': -0.09217800000000001, 's1': -0.13489299999999999},
           {'p2': -0.209385, 'p3': 8.499999999999999e-05, 'p1': -0.152728, 'r0': 0.20001000000000002, 'r1': 0.20305700000000002, 'r2': 0.276104, 'r3': 0.21324300000000002, 's3': -0.020113, 's2': -0.109978, 's1': -0.10892500000000001},
           {'p2': -0.21046499999999999, 'p3': -0.042651999999999995, 'p1': -0.162243, 'r0': 0.20001000000000002, 'r1': 0.20001000000000002, 'r2': 0.264392, 'r3': 0.231603, 's3': -0.070359, 's2': -0.153367, 's1': -0.1217}]


vectors = [{'r0': 0.203795, 'r1': 0.2, 'r2': 0.2, 'r3': 0.269037, 's3': -0.006079, 's2': -0.00348, 's1': -0.070115},
           {'r0': 0.24906, 'r1': 0.2, 'r2': 0.205319, 'r3': 0.249118, 's3': -0.004591, 's2': -0.000385, 's1': -0.035766},
 {'r0': 0.2, 'r1': 0.2, 'r2': 0.277873, 'r3': 0.280233, 's3': 0.00195, 's2': 0.001345, 's1': -0.032942},
{'r0': 0.24906, 'r1': 0.2, 'r2': 0.306868, 'r3': 0.249118, 's3': 0.000862, 's2': -0.001336, 's1': -0.048956},
{'r0': 0.237235, 'r1': 0.2, 'r2': 0.267718, 'r3': 0.261162, 's3': -0.006669, 's2': 0.004421, 's1': -0.064895},
 {'r0': 0.221288, 'r1': 0.2, 'r2': 0.223155, 'r3': 0.296759, 's3': -0.002383, 's2': 0.004997, 's1': -0.029875},
 {'r0': 0.233377, 'r1': 0.2, 'r2': 0.243644, 'r3': 0.261162, 's3': -0.004684, 's2': 0.004804, 's1': -0.045711},
 {'r0': 0.2, 'r1': 0.2, 'r2': 0.270556, 'r3': 0.276756, 's3': -0.004591, 's2': -0.000385, 's1': -0.045909},
 {'r0': 0.224803, 'r1': 0.2, 'r2': 0.2, 'r3': 0.261162, 's3': -0.006669, 's2': 0.003713, 's1': -0.045961},
 {'r0': 0.211601, 'r1': 0.2, 'r2': 0.308447, 'r3': 0.249118, 's3': 0.000862, 's2': -0.002136, 's1': -0.046039}]
key_map = {}
key_map["ng0"] = "max"
key_map["bandwidth"] = "max"
key_map["delay"] = "max"

pareto_function = ParetoMaxFunction(experiment, key_map)

solutions = []
print "Evaluating PCW"

for vec in vectors:
    pcw = PhCWDesign(vec, 0, constraintFunctions)
    solutions.append(pcw)
    pareto_function.evaluate(pcw)
    print str(pcw.solution_vector) + "\n"
    print str(pcw.figures_of_merit) + "\n"


print "generating latex table"

print "\hline \n & Tunable Delay      & Group Index & Bandwidth   & r0       & r1       & r2       & r3    & s1       & s2       & s3       \\\\  \n \hline"
for pc in solutions:
    for k in pc.solution_vector.keys():
        pc.solution_vector[k] = float("{0:.4f}".format(pc.solution_vector[k]))

    print_str = " & " + str(pc.figures_of_merit["delay"]) +" & " + str(pc.figures_of_merit["ng0"]) + " & " + str(pc.figures_of_merit["bandwidth"])
    print_str += " & " + str(pc.solution_vector["r0"]) + " & " + str(pc.solution_vector["r1"]) + " & " + str(pc.solution_vector["r2"])
    print_str += " & " + str(pc.solution_vector["r3"])
    #" & " + str(pc.solution_vector["p1"]) + " & " + str(pc.solution_vector["p2"]) + " & " + str(pc.solution_vector["p3"])
    print_str += " & " + str(pc.solution_vector["s1"]) + " & " + str(pc.solution_vector["s2"])
    print_str += " & " + str(pc.solution_vector["s3"]) + "\\\\ \n \hline"
    print print_str
