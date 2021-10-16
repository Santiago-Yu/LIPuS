import torch
from torch import nn

from PT_generators.RL_Prunning.Conifg import config

from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


class PolicyNetwork(nn.Module):
    def __init__(self, ptg, func):
        super().__init__()
        self.layer = nn.Linear(config.SIZE_EXP_NODE_FEATURE*3,config.SIZE_EXP_NODE_FEATURE)
        self.ptg = ptg
        self.func = func

    def forward(self, stateVec, overall_feature):
        programFearture = self.func(self.ptg.path2CFile, self.ptg.depth)
        l1out = self.layer(torch.cat([stateVec, overall_feature, programFearture], 1))
        return l1out


    def GetParameters(self):
        res = {}
        PreFix = "PolicyNetwork_P_"
        res.update(getParFromModule(self.layer, prefix=PreFix + "layer"))
        return res

    def cudalize(self):
        self.layer = self.layer.cuda()