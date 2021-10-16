import torch
from torch import nn, tensor

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings


class DistributionLize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, action_vector, available_acts):
        #construt the available action vectors
        rawness = torch.cat([torch.mm(SymbolEmbeddings[str(x)], action_vector.transpose(0,1)) for x in available_acts],1)
        likenesses = torch.softmax(rawness,1)
        return (likenesses, rawness)

    def GetParameters(self):
        res = {}

        return res



