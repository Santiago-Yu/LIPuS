from z3 import z3


def extractExpSMT(smtContent):
    part1 = smtContent.split('define-fun')[0]
    part2 = smtContent.split(') Bool')[-1]
    smtContent2 = part1 + ' assert ' + part2

    decl = z3.parse_smt2_string(smtContent2)
    return decl[0]




def GetThree_SMT_conditions_from_SMT_file(path2SMT):
    content = open(path2SMT, "r").read()
    decVars = content.split('( define-fun inv-f')[0]

    decPre = content[content.find('( define-fun pre-f'): content.find('( define-fun trans-f')]

    decTrans = content[content.find('( define-fun trans-f'): content.find('( define-fun post-f')]

    decPost = content[content.find('( define-fun post-f'):]
    decPost = decPost[:decPost.find('SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop')]

    pre_exp = extractExpSMT(decVars + '\n' +  decPre)
    trans_exp = extractExpSMT(decVars + '\n' + decTrans)
    post_exp = extractExpSMT(decVars + '\n' + decPost)
    return pre_exp, trans_exp, post_exp


def parseSMT(path2SMT):
    return GetThree_SMT_conditions_from_SMT_file(path2SMT)