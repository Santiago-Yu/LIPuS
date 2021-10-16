import torch
from torch import nn, tensor

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


class IntLize(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = nn.Linear(SIZE_EXP_NODE_FEATURE, SIZE_EXP_NODE_FEATURE//2)
        # self.layer2 = nn.Linear(SIZE_EXP_NODE_FEATURE//2, 1)
        self.layer1_n = nn.Linear(config.SIZE_EXP_NODE_FEATURE, config.SIZE_EXP_NODE_FEATURE//2)
        self.layer2_n = nn.Linear(config.SIZE_EXP_NODE_FEATURE//2, 1)

    def forward(self, action_vector, left_handle):
        # print("value", self.layer2(self.layer1(action_vector)))
        if str(left_handle.decl()) == 'non_nc' or str(left_handle.decl()) == 'non_nd':
            return torch.min(
                torch.cat([torch.max(
                    torch.cat([self.layer2_n(self.layer1_n(action_vector)),
                               tensor([[1]],dtype=torch.float32)],0)).reshape(1,1),
                           tensor([[4]],dtype=torch.float32)],0))
        assert False # should not be here now
        # else:
        #     return self.layer2(self.layer1(action_vector))


    def GetParameters(self):
        res = {}
        PreFix = "IntLize_P_"

        # res.update(getParFromModule(self.layer1, prefix=PreFix + "layer1"))
        # res.update(getParFromModule(self.layer2, prefix=PreFix + "layer2"))
        res.update(getParFromModule(self.layer1_n, prefix=PreFix + "layer1_n"))
        res.update(getParFromModule(self.layer2_n, prefix=PreFix + "layer2_n"))
        return res




