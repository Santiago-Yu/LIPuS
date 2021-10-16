import torch
from torch import nn, tensor

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(config.SIZE_EXP_NODE_FEATURE * 2, config.SIZE_EXP_NODE_FEATURE)
        self.layer2 = nn.Linear(config.SIZE_EXP_NODE_FEATURE, config.SIZE_EXP_NODE_FEATURE // 2)
        self.layer3 = nn.Linear(config.SIZE_EXP_NODE_FEATURE // 2, 1)

    def forward(self, stateVec, overall_feature):
        tensorflow = tensor(torch.cat([stateVec, overall_feature], 1))
        if torch.cuda.is_available():
            tensorflow = tensorflow.cuda()
        l1out = self.layer1(tensorflow)
        m10 = tensor([[-10]])
        p10 = tensor([[10]])
        if torch.cuda.is_available():
            m10 = m10.cuda()
            p10 = p10.cuda()
        return torch.min(torch.cat([torch.max(torch.cat([self.layer3(self.layer2(l1out)), m10], 1)).reshape(1,1), p10], 1)).reshape(1,1)

    def GetParameters(self):
        res = {}
        PreFix = "RewardPredictor_P_"
        res.update(getParFromModule(self.layer1, prefix=PreFix + "layer1"))
        res.update(getParFromModule(self.layer2, prefix=PreFix + "layer2"))
        res.update(getParFromModule(self.layer3, prefix=PreFix + "layer3"))

        return res

    def cudalize(self):
        self.layer1 = self.layer1.cuda()
        self.layer2 = self.layer2.cuda()
        self.layer3 = self.layer3.cuda()


# little test

if __name__ == "__main__":
    stateVec = torch.randn([1, config.SIZE_EXP_NODE_FEATURE])
    overall_feature = torch.randn([1, config.SIZE_EXP_NODE_FEATURE])
    rp = RewardPredictor()
    print(rp(stateVec, overall_feature))
