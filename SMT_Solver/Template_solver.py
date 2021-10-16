import random
import signal

from z3 import *
set_param('parallel.enable', True)
from SMT_Solver.Config import config
from Utilities.SMT_parser import getConstsFromZ3Exp
from Utilities.TimeController import time_limit_calling


def Substitute(PT, assignment):
    assignedVars = assignment.keys()
    canI = PT
    for avar in assignedVars:
        # try:
        theValue = IntVal(str(assignment[avar]))
        avar = Int(str(avar))
        canI = z3.z3.substitute(canI, (avar, theValue))
        # except Exception as e:
        #     print(e)

    return canI #remember



def solve(PT, CE):

    sol = z3.SolverFor("QF_NIA")
    sol.set(auto_config=False)
    sol.set("timeout", config.PT_SOLVING_TIME)


    # Substitute all program vars with CE table.
    Query = And(True,True)
    P_sampled = CE['p'] if len(CE['p']) <= config.PT_SOLVING_MAX_CE else random.sample(CE['p'], config.PT_SOLVING_MAX_CE)
    N_sampled = CE['n'] if len(CE['n']) <= config.PT_SOLVING_MAX_CE else random.sample(CE['n'], config.PT_SOLVING_MAX_CE)
    I_sampled = CE['i'] if len(CE['i']) <= config.PT_SOLVING_MAX_CE else random.sample(CE['i'], config.PT_SOLVING_MAX_CE)

    for counterexample in P_sampled:
        pterm = Substitute(PT, counterexample)
        Query = And(Query, pterm)
    for counterexample in N_sampled:
        pterm = Substitute(PT, counterexample)
        Query = And(Query, Not(pterm))
    for counterexample in I_sampled:
        pre = Substitute(PT, counterexample[0])
        post = Substitute(PT, counterexample[1])
        Query = And(Query, Implies(pre, post))

    # Try to find a solution.
    try:
        Query = simplify(Query)
    except Exception as e:
        print(e)
    sol.reset()
    # set to QFNIA
    sol.add(Query)



    r = time_limit_calling(sol.check, (Query), config.PT_SOLVING_TIME)


    if r == z3.sat: # coool
        m = sol.model()
        assignment = {}
        for s in m:
            if 'const_' not in str(s):
                continue
            assignment[str(s)] = str(m[s])

        consts = getConstsFromZ3Exp(PT)
        for conster in consts:
            if str(conster) not in assignment: # that means the consts can be any value
                 assignment[conster] = IntVal(random.randint(-10,10))

        return Substitute(PT, assignment)

    elif r == z3.unsat: # Not Cool
        return None
    else:
        raise TimeoutError("template solving is OOT:    " + str(PT))




