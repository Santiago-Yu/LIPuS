from z3 import If
class Config:
    SELECT_AN_ACTION = 0
    SET_AN_VALUE = 1

    SIZE_EXP_NODE_FEATURE = 128
    SIZE_PCA_NUM = 50

    MAX_DEPTH = 150
    BEST = False

    CONTINUE_TRAINING = True

    LinearPrograms = ["Problem_L" + str(i) for i in range(1,134)]
    NonLinearPrograms = ["Problem_NL" + str(i) for i in range(1,31)]

    LearningRate = 1e-6

config = Config()

def Z3_abs(x):
    return If(x >= 0,x,-x)