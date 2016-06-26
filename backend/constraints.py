# Sean Billings, 2015

# This module is a workspace to define the set of constraints that can
# be enforced while executing the optimization routine
# constraints are functions that restrict a parameter to a given bounds
# furthermore it is expected that the constraints should map violating values back to the required bounds
# many of the following examples achieve this mapping recursively
import math
import random

#maps negative constraints to 0
def constraint_postive(parameterMap, mapField):
    if parameterMap[mapField] < 0:
        parameterMap[mapField] = 0


def constraintRounding(parameterMap):
    for key in parameterMap.keys():
        parameterMap[key] = float("{0:.6f}".format(parameterMap[key]))


def constraintAP1(parameterMap):
    if "p1" in parameterMap.keys():
        p1 = -parameterMap["p1"]
        if p1 >= 0.5:
            parameterMap["p1"] = -0.5

def constraint0P1(parameterMap):
    if "p1" in parameterMap.keys():
        p1 = -parameterMap["p1"]
        if p1 < -0.5:
            parameterMap["p1"] = 0.5

def constraintAP2(parameterMap):
    if "p2" in parameterMap.keys():
        p2 = -parameterMap["p2"]
        if p2 >= 0.5:
            parameterMap["p2"] = -0.5

def constraint0P2(parameterMap):
    if "p2" in parameterMap.keys():
        p2 = -parameterMap["p2"]
        if p2 < -0.5:
            parameterMap["p2"] = 0.5

def constraintAP3(parameterMap):
    if "p3" in parameterMap.keys():
        p3 = -parameterMap["p3"]
        if p3 >= 0.5:
            parameterMap["p3"] = -0.5

def constraint0P3(parameterMap):
    if "p3" in parameterMap.keys():
        p3 = -parameterMap["p3"]
        if p3 < -0.5:
            parameterMap["p3"] = 0.5

def constraintAS1(parameterMap):
    if "s1" in parameterMap.keys():
        s1 = -parameterMap["s1"]
        if s1 >= 0.5:
            parameterMap["s1"] = -0.5

def constraint0S1(parameterMap):
    if "s1" in parameterMap.keys():
        s1 = -parameterMap["s1"]
        if s1 < -0.5:
            parameterMap["s1"] = 0.5

def constraintAS2(parameterMap):
    if "s2" in parameterMap.keys():
        s2 = -parameterMap["s2"]
        if s2 >= 0.5:
            parameterMap["s2"] = -0.5

def constraint0S2(parameterMap):
    if "s2" in parameterMap.keys():
        s2 = -parameterMap["s2"]
        if s2 < -0.5:
            parameterMap["s2"] = 0.5

def constraintAS3(parameterMap):
    if "s3" in parameterMap.keys():
        s3 = -parameterMap["s3"]
        if s3 >= 0.5:
            parameterMap["s3"] = -0.5

def constraint0S3(parameterMap):
    if "s3" in parameterMap.keys():
        s3 = -parameterMap["s3"]
        if s3 < -0.5:
            parameterMap["s3"] = 0.5

def constraintAR1(parameterMap):
    if 2*parameterMap["r1"] >= 0.8:
        parameterMap["r1"] = 0.4
    
def constraintAR2(parameterMap):
    if 2*parameterMap["r2"] >= 0.8:
        parameterMap["r2"] = 0.4
        

def constraintAR3(parameterMap):
    if 2*parameterMap["r3"] >= 0.8:
        parameterMap["r3"] = 0.4

def constraint0R1(parameterMap):
    if parameterMap["r1"] < 0.2:
        parameterMap["r1"] = 0.2

def constraint0R2(parameterMap):
    if parameterMap["r2"] < 0.2:
        parameterMap["r2"] = 0.2

def constraint0R3(parameterMap):
    if parameterMap["r3"] < 0.2:
        parameterMap["r3"] = 0.2

def constraintAR0(parameterMap):
    if 2*parameterMap["r0"] >= 0.8:
        parameterMap["r0"] = 0.4

def constraint0R0(parameterMap):
    if parameterMap["r0"] < 0.2:
        parameterMap["r0"] = 0.2


# contrains the overlap between the 4th and 3rd row
def constraintsPSR03(parameterMap):
    if "s3" in parameterMap.keys():
        s3 = -parameterMap["s3"]
    else:
        s3 = 0
    if "p3" in parameterMap.keys():
        p3 = -parameterMap["p3"]
    else:
        p3 = 0
    if "r3" in parameterMap.keys():
        r3 = parameterMap["r3"]
    else:
        r3 = 0
    if "r0" in parameterMap.keys():
        r0 = parameterMap["r0"]
    else:
        r0 = 0

    hole_distance_0_3 = math.sqrt( ((1/math.sqrt(2)) - s3)**2 + ((1/math.sqrt(2)) - p3)**2)

    r0 = parameterMap['r0']
    # r0 = 0.3 # default radius value
    if hole_distance_0_3 - 0.1 < (r3 + r0): # if holes overlap
        # randomly rescale hole radius or shift (unless hole radius is too small)
        if parameterMap["r3"] > 0.23:
            r = random.random()
        else:
            r = 0.6
        if r > 0.5:
            parameterMap['p3'] = 0.9*parameterMap['p3']
            parameterMap['s3'] = 0.9*parameterMap['s3']
            constraintsPSR03(parameterMap)
        else:
            parameterMap['r3'] = 0.9*parameterMap['r3']
            constraintsPSR03(parameterMap)

# contrains the overlap between the 3rd and 2nd row
def constraintsPSR32(parameterMap):
    if "s3" in parameterMap.keys():
        s3 = -parameterMap["s3"]
    else:
        s3 = 0
    if "p3" in parameterMap.keys():
        p3 = -parameterMap["p3"]
    else:
        p3 = 0
    if "s2" in parameterMap.keys():
        s2 = -parameterMap["s2"]
    else:
        s2 = 0
    if "p2" in parameterMap.keys():
        p2 = -parameterMap["p2"]
    else:
        p2 = 0
    if "r3" in parameterMap.keys():
        r3 = parameterMap["r3"]
    else:
        r3 = 0
    if "r2" in parameterMap.keys():
        r2 = parameterMap["r2"]
    else:
        r2 = 0

    hole_distance_3_2 = math.sqrt( ((1/math.sqrt(2)) + s3 - s2)**2 + ((1/math.sqrt(2)) - math.fabs(p3 - p2))**2)

    if hole_distance_3_2 - 0.1 < (r3 + r2 ): # if holes overlap
         # randomly rescale hole radius or shift (unless hole radius is too small)
        if r2 > 0.23:
            r = random.random()
        else:
            r = 0.6

        if r > 0.5:
            if p2 != 0:
                if p2 > p3:
                    parameterMap["p2"] = 0.95*parameterMap["p2"]
                else:
                    parameterMap["p2"] = 1.05*parameterMap["p2"]

            if s2 != 0:
                if s2 > 0:
                    parameterMap['s2'] = 0.9*parameterMap['s2']
                else:
                   parameterMap['s2'] = 1.1*parameterMap['s2']

            constraintsPSR32(parameterMap)
        else:
            parameterMap['r2'] = 0.9*parameterMap['r2']
            constraintsPSR32(parameterMap)

# contrains the overlap between the 2nd and 1st row
def constraintsPSR21(parameterMap):
    if "s2" in parameterMap.keys():
        s2 = -parameterMap["s2"]
    else:
        s2 = 0
    if "p2" in parameterMap.keys():
        p2 = -parameterMap["p2"]
    else:
        p2 = 0
    if "s1" in parameterMap.keys():
        s1 = -parameterMap["s1"]
    else:
        s1 = 0
    if "p1" in parameterMap.keys():
        p1 = -parameterMap["p1"]
    else:
        p1 = 0
    if "r2" in parameterMap.keys():
        r2 = parameterMap["r2"]
    else:
        r3 = 0
    if "r1" in parameterMap.keys():
        r1 = parameterMap["r1"]
    else:
        r1 = 0

    hole_distance_3_2 = math.sqrt( ((1/math.sqrt(2)) + s2 - s1)**2 + ((1/math.sqrt(2)) - math.fabs(p2 - p1))**2)

    if hole_distance_3_2 - 0.1 < (r2 + r1 ): # if holes overlap
         # randomly rescale hole radius or shift (unless hole radius is too small)
        if r1 > 0.23:
            r = random.random()
        else:
            r = 0.6

        if r > 0.5:
            if p1 != 0:
                if p1 > p2:
                    parameterMap["p1"] = 0.95*parameterMap["p1"]
                else:
                    parameterMap["p1"] = 1.05*parameterMap["p1"]

            if s2 != 0:
                if s2 > 0:
                    parameterMap['s1'] = 0.9*parameterMap['s1']
                else:
                   parameterMap['s1'] = 1.1*parameterMap['s1']

            constraintsPSR32(parameterMap)
        else:
            if r1 != 0:
                parameterMap['r1'] = 0.9*parameterMap['r1']
            constraintsPSR32(parameterMap)

# ensures the structural constraints are satisfied for a Line Defect waveguide
def latticeConstraintsLD(parameterMap):

    # print parameterMap
    constraintAR1(parameterMap)
    constraintAR2(parameterMap)
    constraintAR3(parameterMap)

    constraint0R1(parameterMap)
    constraint0R2(parameterMap)
    constraint0R3(parameterMap)


    constraint0R0(parameterMap)
    constraintAR0(parameterMap)
    

    constraintAP1(parameterMap)
    constraintAP2(parameterMap)
    constraintAP3(parameterMap)

    constraint0P1(parameterMap)
    constraint0P2(parameterMap)
    constraint0P3(parameterMap)

    constraintAS1(parameterMap)
    constraintAS2(parameterMap)
    constraintAS3(parameterMap)

    constraint0S1(parameterMap)
    constraint0S2(parameterMap)
    constraint0S3(parameterMap)

    constraintRounding(parameterMap)
    constraintsPSR03(parameterMap)
    constraintsPSR32(parameterMap)
    constraintsPSR21(parameterMap)

# applies each constraint function in 'constraints' to the hashMapVector
# example hashMapVector
# {'p2': 0.07646386156785503, 'p3': 0.043493821552317305, 'p1': 0.07235772695508993,
# 'r1': 0.017137050399521098, 'r2': 0.2053715543744835, 'r3': 0.03296599949382803,
# 's3': 0.011459725712917944, 's2': 0.09901968783066335, 's1': 0.17175319110067366}
def fix(hashMapVector, constraints):
    for con in constraints:
        con(hashMapVector)
    return hashMapVector
