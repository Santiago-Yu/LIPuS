import re
from subprocess import run
import json
from collections import deque

from code2inv.common.cmd_args import cmd_args
from code2inv.common.constants import RULESET

OPERATORS = {'+', '-', '*', '/', '(', ')', '@', '<', '#', '>', '!', '='}
PRIORITY = {'+': 1, '-': 1, '*': 2, '/': 2}
class Stack(object):

    def __init__(self):
        self.__list = []

    def is_empty(self):
        return self.__list == []
    def push(self,item):
        self.__list.append(item)

    def pop(self):
        if self.is_empty():
            return
        else:
            return self.__list.pop()

    def top(self):
        if self.is_empty():
            return
        else:
            return self.__list[-1]

def infix_to_prefix(formula):
    op_stack = []
    exp_stack = []
    for ch in formula:
        if not ch in OPERATORS:
            exp_stack.append(ch)
        elif ch == '(':
            op_stack.append(ch)
        elif ch == ')':
            while op_stack[-1] != '(':
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append(" ".join(["(", op, b, a, ")"]))
            op_stack.pop()  # pop '('
        else:
            while op_stack and op_stack[-1] != '(' and PRIORITY[ch] <= PRIORITY[op_stack[-1]]:
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append(" ".join([op, b, a]))
            op_stack.append(ch)

    # leftover
    while op_stack:
        op = op_stack.pop()
        a = exp_stack.pop()
        b = exp_stack.pop()
        exp_stack.append(" ".join([op, b, a]))
    return exp_stack[-1]


def clean_up(line, paren=False):
    left = ""
    right = ""
    if paren:
        left = "( "
        right = " )"

    clean = line.split('(')[1:]
    clean = left.join(clean)
    clean = clean.split(')')[:-1]
    clean = right.join(clean)

    return clean.strip()


def full_prefix(line):
    clean = line
    if "(" in line:
        clean = clean_up(line, paren=True)

    encode = re.sub(r'<=', '@', clean)
    encode = re.sub(r'>=', '#', encode)
    encode = re.sub(r'!=', '!', encode)
    encode = re.sub(r'==', '=', encode)

    prefix = infix_to_prefix(encode.split())
    decode = re.sub(r'@', '<=', prefix)
    decode = re.sub(r'#', '>=', decode)
    decode = re.sub(r'!', '!=', decode)

    if decode.strip()[0] == "(":
        decode = clean_up(decode, paren=True)

    return decode


def op_conversion(l,noter = False):
    if noter:
        if 'not' in l or 'or' in l or 'and' in l: # has to recurse
            #todo
            print("Juses! recusive op_convension!!!!!")
            pass
        elif '==' in l:
            l = re.sub(r'==', '!=', l)
        elif "!=" in l:
            l = re.sub(r'!=', '==', l)
        elif '>' in l:
            l = re.sub(r'>', '<', l)
        elif '<' in l:
            l = re.sub(r'<', '>', l)
        elif ">=" in l:
            l = re.sub(r'>=', '<=', l)
        elif "<=" in l:
            l = re.sub(r'<=', '>=', l)

    if '==' in l:
        l = re.sub(r'==', '=', l)
    if "!=" in l:
        #l = "not (" + re.sub(r'!=', '=', l) + ")"
        l = "or (" + re.sub(r'!=', '>', l) + ") (" + re.sub(r'!=', '<', l) + ")"
    return l


def to_prefix(l, line):
    if len(l.split()) != 3:
        out = full_prefix(line)
        # print("WARNING: MORE THAN 3 TOKENS",out)
    else:
        out = full_prefix(l)
    return out
def parse_file_conditions(fname):
    with open(fname, 'rt') as input_file:
        preconditions = []
        predicate = None

        post_conditions = {'ifs' :[], 'assert' :None}

        before_loop = True
        post_condition = False
        ifs = []

        for line in input_file:
            if "while" in line:
                before_loop = False
                if ("unknown" in line):
                    continue
                l = clean_up(line)
                l = to_prefix(l ,line)

                l = op_conversion(l)

                predicate = "( " + l + " )"

            if before_loop:
                if "assume" in line:
                    l = clean_up(line)

                    l = to_prefix(l ,line)
                    l = op_conversion(l)
                    preconditions.append('(' + l + ')')
                elif '=' in line:

                    line = line.strip()
                    line = line.strip(';')
                    l = line.strip('()')

                    if l.split()[0] == 'int':

                        l = l.split()
                        l = " ".join(l[1:])

                    l = to_prefix(l ,line)
                    l = op_conversion(l)
                    preconditions.append('(' + l + ')')

            if "post-condition" in line:
                post_condition = True


            if '//' in line:
                continue
            if post_condition and "if" in line and post_condition == True:
                l = clean_up(line)
                l = to_prefix(l ,line)

                l = op_conversion(l, noter=True)  # we try to reverse it here
                ifs.append('(' + l + ')')
                #ifs.append("( not ( "+ l +"))")
            if 'assert' in line:
                l = clean_up(line)
                l = to_prefix(l, line)
                l = " ".join(l.split())
                l = op_conversion(l)
                post_conditions['ifs'] = ifs
                post_conditions['assert'] = "( " + l + " )"

        conditions = {'preconds': preconditions, 'predicate': predicate,
                      'postcondition': post_conditions}
        return conditions


def expHeristicGenerater(sourceFile): # This function produce a static string of expression (fast checker) for cold start. calculating the loss would be hard to calculate.
    # generate simple templates

    condition = parse_file_conditions(sourceFile)

    collection = []
    collection = collection + (condition['preconds'])
    collection.append(condition['predicate'])
    collection.append(condition['postcondition']['assert'])
    collection = collection + (condition['postcondition']['ifs'])
    if condition['predicate']:
        collection.append("(or " + condition['predicate'] + " " + condition['postcondition']['assert'] + ")")
    collection = [i for i in collection if i is not None]
    ands = ["(and " + i + " " + j + " )" for i in collection for j in collection]
    ors = ["(or " + i + " " + j + " )" for i in collection for j in collection]
    collection = collection + ands + ors
    #Debug
    if cmd_args.debug:
        print(collection)
        print(len(collection))
    return collection


class expTree:
    def __init__(self):
        self.children = []
        self.str = ''


def choose(ruler, keypha, keyphaN):
    if keyphaN == -1:
        keyphaN = 0
    rules = RULESET[ruler]
    for i, rule in enumerate(rules):
        counter = 0
        for par in rule:
            if par == keypha:
                counter +=1
        if counter == keyphaN:
            return i
    return -1


def getAndOr(coll):
    stacker = Stack()
    treeStacker = Stack()
    ps = []

    for charer in coll:
        if charer == ')':
            strtmp = ""
            while stacker.top() != '(':
                strtmp += stacker.pop()
            stacker.pop()

            tmp_tree = expTree()
            tmp_tree.str = strtmp[::-1].strip(' ')
            if tmp_tree.str == 'and' or tmp_tree.str == 'or':
                childTreee2 = treeStacker.pop()
                childTreee1 = treeStacker.pop()

                if childTreee1.str == tmp_tree.str:
                    tmp_tree.children.extend(childTreee1.children)
                else:
                    tmp_tree.children.append(childTreee1)

                if childTreee2.str == tmp_tree.str:
                    tmp_tree.children.extend(childTreee2.children)
                else:
                    tmp_tree.children.append(childTreee2)

                treeStacker.push(tmp_tree)

            else:
                ps.append(tmp_tree.str)
                treeStacker.push(tmp_tree)
        else: stacker.push(charer)

    while stacker.top() == ' ':
        stacker.pop()

    root_Tree = treeStacker.pop()
    if not (stacker.is_empty() and treeStacker.is_empty()):
        if cmd_args.debug:
            print("No action for:", coll)
        return None, None

    andOractionSequnce_of_Sequnce = []
    andOractionSequnce = []
    # genearte actions

    if root_Tree.str == 'and':
        #chose the && action
        action = choose('S', '&&', len(root_Tree.children)-1)
        if action == -1:
            if cmd_args.debug:
                print("No action for:", coll)
            return None, None
        andOractionSequnce.append(action)

        for child in root_Tree.children:
            # chose the || action
            action = choose('C', '||', len(child.children) - 1)
            if action == -1:
                if cmd_args.debug:
                    print("No action for:", coll)
                return None, None
            andOractionSequnce.append(action)
            andOractionSequnce_of_Sequnce.append(andOractionSequnce)
            andOractionSequnce = []
            numberOfP = len(child.children)
            while numberOfP > 1:
                numberOfP -= 1
                andOractionSequnce_of_Sequnce.append(andOractionSequnce)
                andOractionSequnce = []


    elif root_Tree.str == 'or':
        # chose the no && action
        action = choose('S', '&&', 0)
        if action == -1:
            if cmd_args.debug:
                print("No action for:   ", coll)
            return None, None
        andOractionSequnce.append(action)

        # chose the || action
        action = choose('C', '||', len(root_Tree.children) - 1)
        if action == -1:
            if cmd_args.debug:
                print("No action for:   ", coll)
            return None, None
        andOractionSequnce.append(action)

        andOractionSequnce_of_Sequnce.append(andOractionSequnce)
        andOractionSequnce = []
        numberOfP = len(root_Tree.children)
        while numberOfP > 1:
            numberOfP -= 1
            andOractionSequnce_of_Sequnce.append(andOractionSequnce)
            andOractionSequnce = []

    else:
        # chose the no && action
        action = choose('S', '&&', 0)
        if action == -1:
            if cmd_args.debug:
                print("No action for:   ", coll)
            return None, None
        andOractionSequnce.append(action)

        # chose the no || action
        action = choose('C', '||', 0)
        if action == -1:
            if cmd_args.debug:
                print("No action for:   ", coll)
            return None, None
        andOractionSequnce.append(action)
        andOractionSequnce_of_Sequnce.append(andOractionSequnce)
        andOractionSequnce = []

    return andOractionSequnce_of_Sequnce, ps




def BothNotIn(cur, variable_dict, const_dict):
    return cur not in variable_dict and cur not in const_dict


def spliting(p):
    res = []
    L = 0
    tmp = ""
    for chare in p:
        if chare == '(':
            L += 1

        elif chare == ')':
            L -= 1
            if L == 0:
                if tmp != "":
                    res.append(tmp)
                tmp = ""

        elif L == 0:
            if chare == ' ':
                if tmp != "":
                    res.append(tmp)
                tmp = ""
            else:
                tmp += chare
        else:
            tmp += chare

    if tmp != "":
        res.append(tmp)
    return res


def askVarOrConst(o1, variable_dict, const_dict):
    if o1 in variable_dict:
        return 0
    elif o1 in const_dict:
        return 1
    return 2


def choose_seq(ruler, appseq):
    rules = RULESET[ruler]
    appi = 0
    for i, rule in enumerate(rules):
        for par in rule:
            if par == appseq[appi]:
                appi += 1
                if appi == len(appseq):
                    break
        if appi == len(appseq):
            return i


def getExpr(p, variable_dict, const_dict):
    actions = []
    if p in variable_dict:
        # choose var
        for i,rule in enumerate(RULESET['expr']):
            if len(rule) == 1 and rule[0] == 'var':
                actions.append(i)
                actions.append(variable_dict[p])
                return actions
        if cmd_args.debug:
            print("No action for expr:=var!!")
        return None

    elif p in const_dict:
        # choose const
        for i, rule in enumerate(RULESET['expr']):
            if len(rule) == 1 and rule[0] == 'const':
                actions.append(i)
                actions.append(const_dict[p])
                return actions
        if cmd_args.debug:
            print("No action for expr:=const!!")
        return None

    else:
        tmp = spliting(p)
        if len(tmp) != 3:
            if cmd_args.debug:
                print("cannot parse:    ", p)
            return None
        oper = tmp[0]
        o1 = tmp[1]
        o2 = tmp[2]
        varOrConst1 = askVarOrConst(o1, variable_dict, const_dict)
        varOrConst2 = askVarOrConst(o2, variable_dict, const_dict)

        if varOrConst1 == varOrConst2 == 0: #2 var
            actions.append(choose('expr', 'var', 2))

            actions.append(variable_dict[o1])
            actions.append(choose('op', oper, 1))
            actions.append(variable_dict[o2])

        elif varOrConst1 == varOrConst2 == 1: #2 const
            actions.append(choose('expr', 'const', 2))

            actions.append(const_dict[o1])
            actions.append(choose('op', oper, 1))
            actions.append(const_dict[o2])

        elif varOrConst1 == 0 and varOrConst2 == 1:
            actions.append(choose_seq('expr', ['var', 'op', 'const']))

            actions.append(variable_dict[o1])
            actions.append(choose('op', oper, 1))
            actions.append(const_dict[o2])

        elif varOrConst1 == 1 and varOrConst2 == 0:
            actions.append(choose_seq('expr', ['const', 'op', 'var']))

            actions.append(const_dict[o1])
            actions.append(choose('op', oper, 1))
            actions.append(variable_dict[o2])


        else:
            if cmd_args.debug:
                print("cannot parse:    ", p)
            return None







def getP(p, variable_dict, const_dict):
    tmp = spliting(p)
    if len(tmp) != 3:
        # Problem occur
        if cmd_args.debug:
            print("cannot choose for    ", p)
        return None
    cmp = tmp[0]
    if cmp == '=':
        cmp = "=="
    o1 = tmp[1] # let it be var
    o2 = tmp[2]

    check1 = BothNotIn(o1, variable_dict, const_dict)
    check2 = BothNotIn(o2, variable_dict, const_dict)

    if check1 and check2:
        # Problem occur
        if cmd_args.debug:
            print("cannot choose for    ", p)
        return None



    if check1: # meaning o2 must be the var, lets swtich it.
        tmmp = o1
        o1 = o2
        o2 = tmmp

        if cmp == '<':
            cmp = '>'
        elif cmp == '>':
            cmp = '<'
        elif cmp == ">=":
            cmp = "<="
        elif cmp == "<=":
            cmp = ">="



    if o1 not in variable_dict:
        # Problem occur
        if cmd_args.debug:
            print("cannot choose for    ", p)
        return None

    # Choose Var
    actions = []
    actions.append(variable_dict[o1])

    # Choose Cmp
    aid = choose('cmp', cmp, 1)
    if aid == -1:
        # Problem occur
        if cmd_args.debug:
            print("cannot choose for    ", p)
        return None

    actions.append(aid)

    # Chose Expr
    eacts = getExpr(o2, variable_dict, const_dict)
    if eacts is None:
        return None

    actions.extend(eacts)
    return actions





def parseIntoAction(collection, programGraph):

    variable_dict = {}
    const_dict = {}
    for i, var in enumerate(programGraph.pg.raw_variable_nodes):
        variable_dict[var] = i

    for j, conster in enumerate(programGraph.pg.const_nodes):
        const_dict[conster] = j
    if cmd_args.debug:
        print(RULESET)

    queuer = []
    for coll in collection:

        actionForOne = []
        andOractionSequnce_Of_Sequnce, ps = getAndOr(coll)
        if andOractionSequnce_Of_Sequnce is None:
            continue
        assert (len(andOractionSequnce_Of_Sequnce) == len(ps))
        #first = True
        breaker = False
        for indexer,p in enumerate(ps):
            actions = getP(p, variable_dict, const_dict)
            if actions is None:
                breaker = True
                break

            actionForOne.extend(actions)
            actionForOne.extend(andOractionSequnce_Of_Sequnce[indexer])

        if breaker:
            continue
        queuer.extend(actionForOne)

    #print(programGraph)

    return deque(queuer)


if __name__ == "__main__":
    expHeristicGenerater(r"G:\reproduce\code2inv-templete-master\benchmarks\C_instances\c\103.c")


