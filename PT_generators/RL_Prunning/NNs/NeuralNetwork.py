import pickle

import torch
from torch import tensor
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.CFG_Embedding import CFG_Embedding
from PT_generators.RL_Prunning.NNs.CounterExampleEmbedding import CEEmbedding
from PT_generators.RL_Prunning.NNs.DistributionLize import DistributionLize
from PT_generators.RL_Prunning.NNs.IntLize import IntLize
from PT_generators.RL_Prunning.NNs.PolicyNetwork import PolicyNetwork
from PT_generators.RL_Prunning.NNs.RewardPredictor import RewardPredictor
from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings
from PT_generators.RL_Prunning.NNs.TreeLSTM import TreeLSTM
from PT_generators.RL_Prunning.TemplateCenter.TemplateCenter import RULE


def constructT():
    treeLSTM = TreeLSTM()
    return treeLSTM

def constructG(cfg):
    return CFG_Embedding(cfg)


def constructE(vars):
    return CEEmbedding(vars)

def constructP():
    return RewardPredictor()

def constructpi(ptg):
    return PolicyNetwork(ptg, GetProgramFearture)

def construct_distributionlize():
    return DistributionLize()

def construct_intValuelzie():
    return IntLize()


def init_symbolEmbeddings():
    Rule_keys =RULE.keys()
    for non_terminal in Rule_keys:
        SymbolEmbeddings[non_terminal] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
        actions = RULE[non_terminal]
        for act in actions:
            SymbolEmbeddings[str(act)] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
    for problems in config.LinearPrograms:
        for depth in range(config.MAX_DEPTH):
            SymbolEmbeddings[problems + "_" + str(depth)] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
    for problems in config.NonLinearPrograms:
        for depth in range(config.MAX_DEPTH):
            SymbolEmbeddings[problems + "_" + str(depth)] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)

def GetProgramFearture(path2CFile, depth):
    problemID = path2CFile.split('/')[-1].split('.')[0]
    if 'NL' in problemID:
        problemStr = "Problem_NL" + problemID.split('NL')[-1]
    else:
        problemStr = "Problem_L" + problemID
    try:
        return SymbolEmbeddings[problemStr + "_" + str(depth)]
    except:
        return SymbolEmbeddings['?']

def GPUlizeSymbols():
    for keyname in SymbolEmbeddings.keys():
        SymbolEmbeddings[keyname] = Parameter(SymbolEmbeddings[keyname].cuda())

def initialize_paramethers(path):
    if "NL" in path:
        ppPath = r"code2inv/templeter/NL_initial.psdlf"
    else:
        ppPath = r"code2inv/templeter/L_initial.psdlf"
    with open(ppPath, 'rb') as f:
        dict = pickle.load(f)
        return dict


def GetActionIndex(last_left_handle,last_action):
    for i, action in enumerate(RULE[str(last_left_handle)]):
        if str(action) == str(last_action):
            if torch.cuda.is_available():
                return tensor([i]).cuda()
            else:
                return tensor([i])

    assert False # should not be here





