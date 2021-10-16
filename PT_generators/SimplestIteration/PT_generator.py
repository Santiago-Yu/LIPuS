# This is a Template Iteration PT generator which try from the simplest.
import random

from Utilities.Cparser import get_varnames_from_source_code
from itertools import combinations, combinations_with_replacement
from z3 import *


class PT_generator:
    s_id = 0

    def OnePass(self, var_names):
        res = []
        expers = []
        for varN in range(1, min(4, len(var_names) + 1)):
            for varlist in combinations(var_names, varN):
                exp = 0
                for varer in varlist:
                    exp += Int(varer) * Int('const_' + str(self.s_id))
                    self.s_id += 1
                res.append(simplify(exp) <= Int('const_' + str(self.s_id)))
                expers.append(simplify(exp))
                self.s_id += 1
        #Non-linear
        res_2 = []
        for exp_two in combinations_with_replacement(expers, 2):
            newExp = exp_two[0] * exp_two[1]
            res_2.append(simplify(newExp) <= Int('const_' + str(self.s_id)))
        res.extend(res_2)
        return res

    def generate_Exp_from_varnames(self, var_names):
        res = []
        res.extend(self.OnePass(var_names))
        return res

    def CNF(self, exps):
        exp_C = True
        for dnormExp in exps:
            exp_d = False
            for exp in dnormExp:
                exp_d = Or(exp_d, exp)
            try:
                exp_d = simplify(exp_d)
            except Exception as e:
                print(e)
            exp_C = And(exp_C, exp_d)
        try:
            exp_C = simplify(exp_C)
        except Exception as e:
            print(e)
        return exp_C

    def __init__(self, path2CFile, path2CFG, path2SMT):
        self.var_names = get_varnames_from_source_code(path2CFile)
        self.exps = self.generate_Exp_from_varnames(self.var_names)
        self.used = set()
        self.lastPT = None

    def generate_next(self, CE):
        for n_c in range(1, 5):
            for n_d in range(1, 3):
                eeee = self.exps[:]
                eeee_i = 0
                if n_c * n_d > len(eeee):
                    print("Nothing to do about it")
                PT = None
                while PT is None or PT in self.used:
                    expser = []
                    if len(eeee) - eeee_i < n_c * n_d:
                        break
                    for x in range(n_c):
                        e_y = []
                        for y in range(n_d):
                            e_y.append(eeee[eeee_i])
                            eeee_i += 1
                        expser.append(e_y)
                    PT = self.CNF(expser)
                if PT in self.used or PT is None:
                    continue
                else:
                    self.lastPT = PT
                    return PT

    def punish(self, SorL, Deg, Whom):
        self.used.add(self.lastPT)

    def prise(self, Deg):
        pass
