import copy
import io
import tokenize

from z3 import z3

from code2inv.common.cmd_args import cmd_args
from code2inv.prog_generator.checkers.c_inv_checker import condense, infix_postfix, postfix_prefix, \
    stringify_prefix_stack

VarEliminated = False
def convertexp2smt(inv):
    inv = inv.replace("&&", "and", -1)
    inv = inv.replace("||", "or", -1)
    b = io.StringIO(inv)
    t = tokenize.generate_tokens(b.readline)
    inv_tokenized = []
    for a in t:
        if a.string != "":
            inv_tokenized.append(a.string)
    inv = stringify_prefix_stack(postfix_prefix(infix_postfix(condense(inv_tokenized))))
    inv = inv.replace("==", "=", -1)
    return inv



def recursivefindconsterandvars(node, vars, consts, varnamesset, lineararg=False):
    if node.rule == "var":
        vars.append(node)
        varnamesset.add(node.name)
        return
    elif node.rule == "const":
        if lineararg:
            node.islinearArg = True
        consts.append(node)
        return
    for child in node.children:
        if node.rule == 'expr':
            recursivefindconsterandvars(child, vars, consts, varnamesset, lineararg=True)
        else:
            recursivefindconsterandvars(child, vars, consts, varnamesset)
    return

def findconsters(expr_root):
    vars = []
    consts = []
    varnamesset = set()
    recursivefindconsterandvars(expr_root, vars, consts, varnamesset)
    return consts, vars, varnamesset


def historydig(expr_root):
    res = ""
    for asserter in expr_root.smthistory:
        res += asserter
    return res

def unsatcore_historydig(expr_root):
    res = ""
    for asserter in expr_root.smthistory:
        res += asserter
    return res, expr_root.varshistory


def z3constsolving(samples, expr_root, negative=False, NotSolve = False):
    sol = z3.Solver()
    sol.set(auto_config=False)
    sol.set("timeout", cmd_args.solTime)

    # '(declare-const x Int) (assert (and (> x 0) (< x 10) (<= x 2)) )'
    consters, vars, varnamesset = findconsters(expr_root)
    smtstr = ""
    constnamesbackup = []
    for i, conster in enumerate(consters):
        tmpconstname = "consBBB_" + str(i)
        # conster.rule = 'var'
        constnamesbackup.append(conster.name)
        conster.name = tmpconstname
        smttmp = '(declare-const ' + tmpconstname + ' Int) '
        smtstr += smttmp
        # we should add the bound of all const
        if conster.islinearArg and cmd_args.octagon:
            smttmp = '(assert (and (<= ' + tmpconstname + ' ' + str(cmd_args.octagon_bound) + ') (>= ' + tmpconstname + ' ' + str(-cmd_args.octagon_bound) + ') ) ) '
        elif cmd_args.const_bound != -1:
            smttmp = '(assert (and (<= ' + tmpconstname + ' ' + str(cmd_args.const_bound) + ') (>= ' + tmpconstname + ' ' + str(-cmd_args.const_bound) + ') ) ) '
        else:
            smttmp = ''
        smtstr += smttmp



    # also need to declare the variables.
    for varname in varnamesset:
        smttmp = '(declare-const ' + varname + ' Int) '
        smtstr += smttmp



    smtstr += historydig(expr_root)

    varnamesbackup = []
    hasvalidcounter = False
    for sample in samples:
        # setvarconfig = set(sample.config.keys())
        # if len(varnamesset - setvarconfig) > 0:
        #     # if len(sample.config) < nofvar:  # should ask kinds map.
        #     continue

        hasvalidcounter = True
        vars_changed = []
        for varer in vars:
            if varer.name in sample.config.keys():
                varnamesbackup.append(varer.name)
                varer.name = sample.config[varer.name]
                vars_changed.append(varer)
        if not negative:
            smttmp = "(assert" + convertexp2smt(str(expr_root)) + ") "

        else:
            smttmp = "(assert (not " + convertexp2smt(str(expr_root)) + ") ) "

        # replace the vares back.
        for i, varer in enumerate(vars_changed):
            varer.name = varnamesbackup[i]
        varnamesbackup.clear()
        vars_changed.clear()

        expr_root.smthistory.append(smttmp)
        smtstr += smttmp

    if not hasvalidcounter or NotSolve:  # do no thing, let it pass
        for i, conster in enumerate(consters): # recover
            conster.name = constnamesbackup[i]
        return 1


    sol.reset()
    # DEBUG
    if cmd_args.debug == True:
        print("smtstr:  ", smtstr)
    # DEBUG
    decl = z3.parse_smt2_string(smtstr)
    sol.add(decl)
    r = sol.check()
    if z3.sat == r:  # find a solution, we need to reform the expr
        # replace the consts wirth solution
        m = sol.model()
        const_sol = {}
        for x in m:
            const_sol[str(x)] = str(m[x])
        for i, conster in enumerate(consters):
            tmpconstname = "consBBB_" + str(i)
            if tmpconstname not in const_sol:
                conster.name = constnamesbackup[i]
            else:
                conster.name = str(const_sol[tmpconstname])
        return 1
    else:
        for i, conster in enumerate(consters): # recover
            conster.name = constnamesbackup[i]
        expr_root.solvePossible = False
        return 0  # consider more detailed back in the future.
def z3check(samples, expr_root, negative=False):
    sol = z3.Solver()
    sol.set(auto_config=False)
    sol.set("timeout", cmd_args.solTime)

    # '(declare-const x Int) (assert (and (> x 0) (< x 10) (<= x 2)) )'
    consters, vars, varnamesset = findconsters(expr_root)
    smtstr = ""

    # need to declare the variables.
    for varname in varnamesset:
        smttmp = '(declare-const ' + varname + ' Int) '
        smtstr += smttmp



    varnamesbackup = []

    for sample in samples:
        vars_changed = []
        for varer in vars:
            if varer.name in sample.config.keys():
                varnamesbackup.append(varer.name)
                varer.name = sample.config[varer.name]
                vars_changed.append(varer)
        if not negative:
            smttmp = "(assert" + convertexp2smt(str(expr_root)) + ") "

        else:
            smttmp = "(assert (not " + convertexp2smt(str(expr_root)) + ") ) "

        # replace the vares back.
        for i, varer in enumerate(vars_changed):
            varer.name = varnamesbackup[i]
        varnamesbackup.clear()
        vars_changed.clear()
        smtstr += smttmp
    sol.reset()
    # DEBUG
    if cmd_args.debug == True:
        print("smtstr:  ", smtstr)
    # DEBUG
    decl = z3.parse_smt2_string(smtstr)
    sol.add(decl)
    r = sol.check()
    if z3.sat == r:  # find a solution, we need to reform the expr
        return 1
    else:
        return 0  # consider more detailed back in the future.

def getsmtstr(config, expr_root, varnamesbackup, vars, notused=None):
    if notused is None:
        notused = []

    not_assigned = []
    not_assigned_i = 0

    for varer in vars:
        # varer.rule = 'const'
        if varer.name in notused:
            not_assigned.append(not_assigned_i)
            continue
        varnamesbackup.append(varer.name)
        varer.name = config[varer.name]
        not_assigned_i += 1
    smttmp = convertexp2smt(str(expr_root))

    varnamesbackup_i = 0
    # replace the vares back.
    for i, varer in enumerate(vars):
        if i not in not_assigned:
            varer.name = varnamesbackup[varnamesbackup_i]
            varnamesbackup_i += 1
    varnamesbackup.clear()
    return smttmp


def I_z3constsolving(samples, expr_root, NotSolve= False):
    sol = z3.Solver()
    sol.set(auto_config=False)
    sol.set("timeout", cmd_args.solTime)

    # '(declare-const x Int) (assert (and (> x 0) (< x 10) (<= x 2)) )'
    consters, vars, varnamesset = findconsters(expr_root)
    smtstr = ""
    constnamesbackup = []
    for i, conster in enumerate(consters):
        tmpconstname = "consBBB_" + str(i)
        # conster.rule = 'var'
        constnamesbackup.append(conster.name)
        conster.name = tmpconstname
        smttmp = '(declare-const ' + tmpconstname + ' Int) '
        smtstr += smttmp
        # we should add the bound of all const
        if conster.islinearArg and cmd_args.octagon:
            smttmp = '(assert (and (<= ' + tmpconstname + ' ' + str(cmd_args.octagon_bound) + ') (>= ' + tmpconstname + ' ' + str(-cmd_args.octagon_bound) + ') ) ) '
        elif cmd_args.const_bound != -1:
            smttmp = '(assert (and (<= ' + tmpconstname + ' ' + str(cmd_args.const_bound) + ') (>= ' + tmpconstname + ' ' + str(-cmd_args.const_bound) + ') ) ) '
        else:
            smttmp = ''
        smtstr += smttmp

    # also need to declare the variables.
    for varname in varnamesset:
        smttmp = '(declare-const ' + varname + ' Int) '
        smtstr += smttmp

    smtstr += historydig(expr_root)

    varnamesbackup = []
    hasvalidcounter = False
    for sample in samples:

        # # check if the sample is full
        # setvarconfig = set(sample.config[0].keys())
        # if len(varnamesset - setvarconfig) > 0:
        #     continue
        hasvalidcounter = True

        var_quntifier = set()  # To find out what variable need to be allized.
        set_corvars = set()
        for varer in sample.config[0].keys():
            set_corvars.add(varer)
        for varer in vars:
            if varer.name not in set_corvars:
                var_quntifier.add(varer.name)

        smtassignment_pre = getsmtstr(sample.config[0], expr_root, varnamesbackup, vars, notused=var_quntifier)
        smtassignment_after = getsmtstr(sample.config[1], expr_root, varnamesbackup, vars, notused=var_quntifier)
        smttmp = "(assert (=> " + smtassignment_pre + " "+ smtassignment_after +") ) "

        expr_root.smthistory.append(smttmp)
        smtstr += smttmp

    if not hasvalidcounter or NotSolve:  # do no thing, let it pass
        for i, conster in enumerate(consters):  # recover
            conster.name = constnamesbackup[i]
        return 1

    sol.reset()

    #DEBUG
    if cmd_args.debug == True:
        print("smtstr:  ", smtstr)
    #DEBUG
    decl = z3.parse_smt2_string(smtstr)
    sol.add(decl)
    r = sol.check()
    if z3.sat == r:  # find a solution, we need to reform the expr
        # replace the consts wirth solution
        m = sol.model()
        const_sol = {}
        for x in m:
            const_sol[str(x)] = str(m[x])
        for i, conster in enumerate(consters):
            tmpconstname = "consBBB_" + str(i)
            if tmpconstname not in const_sol:
                conster.name = constnamesbackup[i]
            else:
                conster.name = str(const_sol[tmpconstname])
        return 1
    else:
        for i, conster in enumerate(consters): # recover
            conster.name = constnamesbackup[i]
        expr_root.solvePossible = False
        return 0  # consider more detailed back in the future.
def I_z3check(samples, expr_root):
    sol = z3.Solver()
    sol.set(auto_config=False)
    sol.set("timeout", cmd_args.solTime)

    # '(declare-const x Int) (assert (and (> x 0) (< x 10) (<= x 2)) )'
    consters, vars, varnamesset = findconsters(expr_root)
    smtstr = ""

    #need to declare the variables.
    for varname in varnamesset:
        smttmp = '(declare-const ' + varname + ' Int) '
        smtstr += smttmp

    varnamesbackup = []

    for sample in samples:

        var_quntifier = set()  # To find out what variable need to be allized.
        set_corvars = set()
        for varer in sample.config[0].keys():
            set_corvars.add(varer)
        for varer in vars:
            if varer.name not in set_corvars:
                var_quntifier.add(varer.name)

        smtassignment_pre = getsmtstr(sample.config[0], expr_root, varnamesbackup, vars, notused=var_quntifier)
        smtassignment_after = getsmtstr(sample.config[1], expr_root, varnamesbackup, vars, notused=var_quntifier)
        smttmp = "(assert (=> " + smtassignment_pre + " "+ smtassignment_after +") ) "
        smtstr += smttmp

    sol.reset()

    #DEBUG
    if cmd_args.debug == True:
        print("smtstr:  ", smtstr)
    #DEBUG
    decl = z3.parse_smt2_string(smtstr)
    sol.add(decl)
    r = sol.check()
    if z3.sat == r:  # find a solution, we need to reform the expr
        return 1
    else:
        return 0  # consider more detailed back in the future.


def CheckOriginalz3(samples, expr_root, sol,  negative):
    sol.reset()
    consters, vars, varnamesset = findconsters(expr_root)
    smtstr = ""
    varnamesbackup = []
    for sample in samples:
        var_quntifier = set() # To find out what variable need to be allized.
        set_corvars = set()
        not_assigned = []
        not_assigned_i = 0
        for varer in sample.config[-1]:
            set_corvars.add(varer)
        for varer in vars:
            if varer.name not in set_corvars:
                var_quntifier.add(varer.name)
                not_assigned.append(not_assigned_i)
            else:
                varnamesbackup.append(varer.name)
                varer.name = sample.config[-2][varer.name]
            not_assigned_i += 1
        smttmp = "(assert " #leftbrace 1
        if len(var_quntifier) > 0: # allize
            #print("Finally used!")
            cmd_args.VarEliminated = True
            smttmp += "(forall (" #leftbrace 2
            for var_q in var_quntifier:
                smttmp += "( " + var_q + " Int ) "
            smttmp += ")"
        if not negative:
            smttmp += convertexp2smt(str(expr_root))
        else:
            smttmp += " (not " + convertexp2smt(str(expr_root)) + ") "

        smttmp += ") "
        if len(var_quntifier) > 0:
            smttmp += ") "
        varnamesbackup_i = 0
        # replace the vares back.
        for i, varer in enumerate(vars):
            if i not in not_assigned:
                varer.name = varnamesbackup[varnamesbackup_i]
                varnamesbackup_i += 1
        varnamesbackup.clear()

        expr_root.smthistory.append(smttmp)
        smtstr += smttmp

    sol.reset()
    # DEBUG
    if cmd_args.debug == True:
        print("smtstr:  ", smtstr)
    # DEBUG
    decl = z3.parse_smt2_string(smtstr)
    sol.add(decl)
    r = sol.check()
    if z3.sat == r:  # find a solution, we need to reform the expr
        return 1.4
    else:
        return 0

def z3unsatcore_constsolving(samples, expr_root, negative=False):  # use 'forall' to give it a try.
    sol = z3.Solver()
    sol.set(auto_config=False)
    sol.set("timeout", cmd_args.solTime)
    oresult = CheckOriginalz3(samples, copy.deepcopy(expr_root),sol,  negative=False)

    # This paper declaration area
    # '(declare-const x Int) (assert (and (> x 0) (< x 10) (<= x 2)) )'
    consters, vars, varnamesset = findconsters(expr_root)
    smtstr = ""
    constnamesbackup = []
    # Before anything happens, we may be better try the original first.



    for i, conster in enumerate(consters):
        tmpconstname = "consBBB_" + str(i)
        # conster.rule = 'var'
        constnamesbackup.append(conster.name)
        conster.name = tmpconstname
        smttmp = '(declare-const ' + tmpconstname + ' Int) '
        smtstr += smttmp

    smtstr += historydig(expr_root)
    varnamesbackup = []

    for sample in samples:


        var_quntifier = set() # To find out what variable need to be allized.
        set_corvars = set()
        not_assigned = []
        not_assigned_i = 0
        for varer in sample.config[-1]:
            set_corvars.add(varer)
        for varer in vars:
            if varer.name not in set_corvars:
                var_quntifier.add(varer.name)
                not_assigned.append(not_assigned_i)
            else:
                varnamesbackup.append(varer.name)
                varer.name = sample.config[-2][varer.name]
            not_assigned_i += 1

        smttmp = "(assert " #leftbrace 1
        if len(var_quntifier) > 0: # allize
            #print("Finally used!")
            cmd_args.VarEliminated = True
            smttmp += "(forall (" #leftbrace 2
            for var_q in var_quntifier:
                smttmp += "( " + var_q + " Int ) "
            smttmp += ")"


        if not negative:
            smttmp += convertexp2smt(str(expr_root))
        else:
            smttmp += " (not " + convertexp2smt(str(expr_root)) + ") "

        smttmp += ") "
        if len(var_quntifier) > 0:
            smttmp += ") "
        varnamesbackup_i = 0
        # replace the vares back.
        for i, varer in enumerate(vars):
            if i not in not_assigned:
                varer.name = varnamesbackup[varnamesbackup_i]
                varnamesbackup_i += 1
        varnamesbackup.clear()

        expr_root.smthistory.append(smttmp)
        smtstr += smttmp

    if oresult > 0:
        if cmd_args.debug == True:
            print("One Shot!")
        for i, conster in enumerate(consters):  # recover
            conster.name = constnamesbackup[i]
        return oresult

    sol.reset()
    # DEBUG
    if cmd_args.debug == True:
        print("smtstr:  ", smtstr)
    # DEBUG
    decl = z3.parse_smt2_string(smtstr)
    sol.add(decl)
    r = sol.check()
    if z3.sat == r:  # find a solution, we need to reform the expr
        # replace the consts wirth solution
        m = sol.model()
        const_sol = {}
        for x in m:
            const_sol[str(x)] = str(m[x])
        for i, conster in enumerate(consters):
            tmpconstname = "consBBB_" + str(i)
            if tmpconstname not in const_sol:
                conster.name = constnamesbackup[i]
            else:
                conster.name = str(const_sol[tmpconstname])
        return 1
    else:
        for i, conster in enumerate(consters):  # recover
            conster.name = constnamesbackup[i]
        expr_root.solvePossible = False
        return 0  # consider more detailed back in the future.


def I_CheckOriginalz3(samples, expr_root, sol):
    sol.reset()
    consters, vars, varnamesset = findconsters(expr_root)
    smtstr = ""
    varnamesbackup = []
    for sample in samples:

        var_quntifier = set()  # To find out what variable need to be allized.
        set_corvars = set()
        for varer in sample.config[-1]:
            set_corvars.add(varer)
        for varer in vars:
            if varer.name not in set_corvars:
                var_quntifier.add(varer.name)

        smtassignment_pre = getsmtstr(sample.config[-2][0], expr_root, varnamesbackup, vars, notused=var_quntifier)
        smtassignment_after = getsmtstr(sample.config[-2][1], expr_root, varnamesbackup, vars, notused=var_quntifier)


        smttmp = "(assert "  # leftbrace 1
        if len(var_quntifier) > 0:  # allize
            #print("Finally used!    I!!!")
            cmd_args.VarEliminated = True
            smttmp += "(forall ("  # leftbrace 2
            for var_q in var_quntifier:
                smttmp += "( " + var_q + " Int ) "
            smttmp += ")"

        smttmp += "(=> "
        smttmp += smtassignment_pre + " "
        smttmp += smtassignment_after + " "
        smttmp += ")"
        smttmp += ") "
        if len(var_quntifier) > 0:
            smttmp += ") "
        #expr_root.smthistory.append(smttmp)
        smtstr += smttmp
    # DEBUG
    if cmd_args.debug == True:
        print("smtstr:  ", smtstr)
    # DEBUG
    decl = z3.parse_smt2_string(smtstr)
    sol.add(decl)
    r = sol.check()
    if z3.sat == r:
        return 1.4
    else:
        return 0
def I_unsatcore_z3constsolving(samples, expr_root):
    sol = z3.Solver()
    sol.set(auto_config=False)
    sol.set("timeout", cmd_args.solTime)
    oresult = I_CheckOriginalz3(samples, copy.deepcopy(expr_root),sol)

    # This paper declaration area
    # '(declare-const x Int) (assert (and (> x 0) (< x 10) (<= x 2)) )'
    consters, vars, varnamesset = findconsters(expr_root)
    smtstr = ""
    constnamesbackup = []
    for i, conster in enumerate(consters):
        tmpconstname = "consBBB_" + str(i)
        # conster.rule = 'var'
        constnamesbackup.append(conster.name)
        conster.name = tmpconstname
        smttmp = '(declare-const ' + tmpconstname + ' Int) '
        smtstr += smttmp



    smtstr += historydig(expr_root)


    varnamesbackup = []
    for sample in samples:

        var_quntifier = set()  # To find out what variable need to be allized.
        set_corvars = set()
        for varer in sample.config[-1]:
            set_corvars.add(varer)
        for varer in vars:
            if varer.name not in set_corvars:
                var_quntifier.add(varer.name)

        smtassignment_pre = getsmtstr(sample.config[-2][0], expr_root, varnamesbackup, vars, notused=var_quntifier)
        smtassignment_after = getsmtstr(sample.config[-2][1], expr_root, varnamesbackup, vars, notused=var_quntifier)


        smttmp = "(assert "  # leftbrace 1
        if len(var_quntifier) > 0:  # allize
            #print("Finally used!    I!!!")
            cmd_args.VarEliminated = True
            smttmp += "(forall ("  # leftbrace 2
            for var_q in var_quntifier:
                smttmp += "( " + var_q + " Int ) "
            smttmp += ")"

        smttmp += "(= "
        smttmp += smtassignment_pre + " "
        smttmp += smtassignment_after + " "
        smttmp += ")"
        smttmp += ") "
        if len(var_quntifier) > 0:
            smttmp += ") "
        expr_root.smthistory.append(smttmp)
        smtstr += smttmp

    if oresult > 0:
        if cmd_args.debug == True:
            print("One Shot!")
        for i, conster in enumerate(consters):  # recover
            conster.name = constnamesbackup[i]
        return oresult

    sol.reset()
    # DEBUG
    if cmd_args.debug == True:
        print("smtstr:  ", smtstr)
    # DEBUG
    decl = z3.parse_smt2_string(smtstr)
    sol.add(decl)
    r = sol.check()
    if z3.sat == r:  # find a solution, we need to reform the expr
        # replace the consts wirth solution
        m = sol.model()
        const_sol = {}
        for x in m:
            const_sol[str(x)] = str(m[x])
        for i, conster in enumerate(consters):
            tmpconstname = "consBBB_" + str(i)
            if tmpconstname not in const_sol:
                conster.name = constnamesbackup[i]
            else:
                conster.name = str(const_sol[tmpconstname])
        return 1
    else:
        for i, conster in enumerate(consters):  # recover
            conster.name = constnamesbackup[i]
        expr_root.solvePossible = False
        return 0  # consider more detailed back in the future.

def const_smt_solve(samples, type, expr_root):
    if not expr_root.solvePossible: # no longer solvable.
        return 0

    if type == 'T:': # meaning solve a true one.
        if not cmd_args.unsat_core:
            return z3constsolving(samples, expr_root)
        else:
            return z3unsatcore_constsolving(samples, expr_root)

    elif type == 'F:':
        if not cmd_args.unsat_core:
            return z3constsolving(samples, expr_root, negative=True)
        else:
            return z3unsatcore_constsolving(samples, expr_root, negative=True)

    elif type == 'I:':
        if not cmd_args.unsat_core:
            return I_z3constsolving(samples, expr_root)
        else:
            return I_unsatcore_z3constsolving(samples, expr_root)

def const_smt_save(samples, type, expr_root):
    if not expr_root.solvePossible: # no longer solvable.
        return 0

    if type == 'T:': # meaning solve a true one.
        return z3constsolving(samples, expr_root, NotSolve = True)

    elif type == 'F:':
        return z3constsolving(samples, expr_root, negative=True, NotSolve = True)

    elif type == 'I:':
        return I_z3constsolving(samples, expr_root, NotSolve = True)

def smt_check(samples, type, expr_root):
    if not expr_root.solvePossible: # no longer solvable.
        return 0
    if type == 'T:': # meaning solve a true one.
        return z3check(samples, expr_root)

    elif type == 'F:':
        return z3check(samples, expr_root, negative=True)

    elif type == 'I:':
        return I_z3check(samples, expr_root)
