import torch
from torch import tensor
from z3 import *

from PT_generators.RL_Prunning.Conifg import config

# RULE = {
#     'non_nc': [And(Bool('non_nd')), And(Bool('non_nd'), Bool('non_nd')), And(Bool('non_nd'), Bool('non_nd'), Bool('non_nd'))],
#     'non_nd': [Or(Bool('non_p')), Or(Bool('non_p'), Bool('non_p')), Or(Bool('non_p'), Bool('non_p'), Bool('non_p'))],
#     'non_p': [Int('non_t') < Int('non_s'),
#               Int('non_t') <= Int('non_s'),
#               Int('non_t') == Int('non_s'),
#               Int('non_t') > Int('non_s'),
#               Int('non_t') >= Int('non_s')],
#     'non_t': [Int('non_v'), Int('non_s'), Int('non_op2')],
#     'non_op2': [Int('non_t') + Int('non_t'), Int('non_t') - Int('non_t'), Int('non_t') * Int('non_t'),
#                 Int('non_t') / Int('non_t'), Int('non_t') % Int('non_t')],
#     #'non_op1': [-Int('non_t')],  # 'Rule_op1_abs'],
#     'non_s': [Int('undecided')], #Int('non_decided')
#     # 'non_decided': ['VALUE'],
#     'non_v': []  # dynamically initialize this one
# }
RULE = {
    'non_nc': [And(Bool('non_nd')), And(Bool('non_nd'), Bool('non_nd')),
               And(Bool('non_nd'), Bool('non_nd'), Bool('non_nd'))],
    'non_nd': [Or(Bool('non_p')), Or(Bool('non_p'), Bool('non_p')), Or(Bool('non_p'), Bool('non_p'), Bool('non_p'))],
    'non_p': [Int('non_t') < Int('non_s'),
              Int('non_t') <= Int('non_s'),
              Int('non_t') == Int('non_s')],
    'non_t': [Int('non_term'),
              Int('non_term') + Int('non_term'),
              Int('non_term') + Int('non_term') + Int('non_term'),
              Int('non_term') + Int('non_term') + Int('non_term') + Int('non_term')],
    'non_term': [Int('non_v'),
                 Int('non_s') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v') * Int('non_v') * Int('non_v')],
    # 'non_op1': [-Int('non_t')],  # 'Rule_op1_abs'],
    'non_s': [Int('undecided')],  # Int('non_decided')
    # 'non_decided': ['VALUE'],
    'non_v': []  # dynamically initialize this one
}
const_ID = 0


def InitPT():
    global const_ID
    const_ID = 0
    return Bool('non_nc')


def getLeftHandle(PT):
    if 'non_' in str(PT.decl()):
        return PT
    else:
        for child in PT.children():
            l = getLeftHandle(child)
            if l is not None:
                return l
    return None


def AvailableActionSelection(left_handle):
    # if len(RULE[str(left_handle.decl())]) == 1 and str(RULE[str(left_handle.decl())][0]) == 'VALUE':
    #     return SET_AN_VALUE, None
    # else:
    return config.SELECT_AN_ACTION, RULE[str(left_handle.decl())]


def init_varSelection(vars):
    RULE['non_v'] = [Int(v) for v in vars]
    SIMPLEST_RULE['non_v'] = [Int(v) for v in vars]


def init_constSelection(consts):
    RULE['non_s'].extend([IntVal(s) for s in consts])


def substitute_the_leftmost_one(node, left_handle, replacer):
    if 'non_' in str(node.decl()):
        assert str(node.decl()) == str(left_handle.decl())  # Since this must be the left most one.
        return True, replacer
    else:
        childs = node.children()
        if len(childs) >= 1:
            newchilds = []
            replaced = False
            for i, child in enumerate(childs):
                replaced, child_after = substitute_the_leftmost_one(child, left_handle, replacer)
                newchilds.append(child_after)
                if replaced:
                    i += 1  # The current i has been used
                    break
            if i < len(childs):
                newchilds.extend(childs[i:])
            if replaced:
                try:
                    return True, getattr(z3, str(node.decl()))(newchilds)
                except:
                    return True, node.decl()(newchilds)
            else:
                return False, node
        else:  # just return
            return False, node


def update_PT_rule_selction(PT, left_handle, action_selected):
    assert str(action_selected) != 'VALUE'
    if str(action_selected) == 'undecided':
        global const_ID
        action_selected = Int('const_' + str(const_ID))
        const_ID += 1
    return substitute_the_leftmost_one(PT, left_handle, action_selected)[1]


def update_PT_value(PT, left_handle, value_of_int):
    if str(left_handle) == 'non_nc':
        return And([Bool('non_nd')] * value_of_int)
    elif str(left_handle) == 'non_nd':
        return substitute_the_leftmost_one(PT, left_handle, Or([Bool('non_p')] * value_of_int))[1]
    else:
        return substitute_the_leftmost_one(PT, left_handle, IntVal(value_of_int))[1]


# def ShouldStrict(lefthandle, Whom):
#     if Whom == "V":
#         if str(lefthandle) in ['non_nc', 'non_nd', 'non_t', 'non_op2']:
#             return True
#         else:
#             return False
#     else:
#         assert Whom == "S"
#         if str(lefthandle) in ['non_nc', 'non_nd', 'non_t', 'non_op2', 'non_s']:
#             return True
#         else:
#             return False

def ShouldStrict(lefthandle, Whom):
    if Whom == "V":
        if str(lefthandle) in ['non_nc', 'non_nd', 'non_t', 'non_term']:
            return True
        else:
            return False
    else:
        assert Whom == "S"

        if str(lefthandle) in ['non_nc', 'non_nd', 'non_t', 'non_term', 'non_s']:
            return True
        else:
            return False


# def StrictnessDirtribution(lefthandle, Whom):
#     if str(lefthandle) == 'non_op2':
#         return tensor([[0.4, 0.4, 0.1, 0.05, 0.05]], dtype=torch.float32)
#
#     if Whom == "V":
#         assert str(lefthandle) == 'non_t'
#         return tensor([[0.1, 0.85, 0, 0.05]], dtype=torch.float32)
#     else:
#         assert Whom == "S"
#         assert str(lefthandle) in ['non_t', 'non_s']
#         if str(lefthandle) == 'non_t':
#             return tensor([[0.85, 0.1, 0, 0.05]], dtype=torch.float32)
#         else:
#             return tensor([[1, 0]], dtype=torch.float32)

def StrictnessDirtribution(lefthandle, Whom):
    distri_dict = {
        'non_nc': [0.95, 0.05, 0.0],
        'non_nd': [0.95, 0.05, 0.0],
        'non_t': [0.95,
                  0.049,
                  0.001,
                  0.0],
        'non_term': [0.5055,
                     0.4944,
                     0.0001,
                     0.0,
                     0.0],
        'non_s': [0]
    }
    if Whom == "S":
        distri_dict['non_term'] = [0.99, 0.01, 0.0, 0.0,0.0]

    if (len(RULE['non_s']) - 1) > 0:
        distri_dict['non_s'].extend([1 / (len(RULE['non_s']) - 1)] * (len(RULE['non_s']) - 1))
    else:
        distri_dict['non_s'] = [1]
    for kk in distri_dict:
        try:
            assert len(distri_dict[kk]) == len(RULE[kk])
        except Exception as e:
            print(e)
            raise e

    res = tensor([distri_dict[str(lefthandle)]], dtype=torch.float32)
    if torch.cuda.is_available():
        res = res.cuda()
    return res


def LossnessDirtribution(lefthandle, Whom): #only S will ask it.
    distri_dict = {
        'non_nc': [0.0, 0.25, 0.75],
        'non_nd': [0.0, 0.25, 0.75],
        'non_t': [0.05,
                  0.15,
                  0.2,
                  0.6],
        'non_term': [0.0,
                     0.8,
                     0.19,
                     0.009,
                     0.001],
        'non_s': [1]
    }
    if (len(RULE['non_s']) - 1) > 0:
        distri_dict['non_s'].extend([0] * (len(RULE['non_s']) - 1))

    for kk in distri_dict:
        try:
            assert len(distri_dict[kk]) == len(RULE[kk])
        except Exception as e:
            print(e)
            raise e

    res = tensor([distri_dict[str(lefthandle)]], dtype=torch.float32)
    if torch.cuda.is_available():
        res = res.cuda()
    return res


# SIMPLEST_RULE = {
#     'non_nc': [And(Bool('non_nd'))],
#     'non_nd': [Or(Bool('non_p'))],
#     'non_p': [Int('non_t') < Int('non_s')],
#     'non_t': [Int('non_s')],
#     'non_op2': [Int('non_t') + Int('non_t')],
#     #'non_op1': [-Int('non_t')],  # 'Rule_op1_abs'],
#     'non_s': [Int('undecided')],
#     #'non_decided': ['VALUE'],
#     'non_v': []  # dynamically initialize this one
# }
SIMPLEST_RULE = {
    'non_nc': [And(Bool('non_nd'))],
    'non_nd': [Or(Bool('non_p'))],
    'non_p': [Int('non_t') < Int('non_s')],
    'non_t': [Int('non_term')],
    'non_term': [Int('non_v')],
    'non_s': [Int('undecided')],
    'non_v': []  # dynamically initialize this one
}


def simplestAction(left_handle):
    return SIMPLEST_RULE[str(left_handle)][0]


# liitel test
if __name__ == "__main__":
    exp = And(Int('x') + Int('y') < 3, Bool('non_p'), Int('non_t') < Int('non_t'))
    print(exp)
    exp = substitute_the_leftmost_one(exp, getLeftHandle(exp), Int('non_t') >= Int('non_t'))[1]
    exp = substitute_the_leftmost_one(exp, getLeftHandle(exp), Int('z') % Int('q'))
    print(exp[1])
