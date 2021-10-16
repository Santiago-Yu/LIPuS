import torch
from torch import nn, tensor
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import *

from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


class TreeLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.RNNS = {}
        # Lets give each sort of Z3 EXP an rnn
        ops = '+,-,*,/,%,If'.split(',')
        cps = '<,>,<=,>=,=='.split(',')
        lcs = 'Not,And,Or,Implies'.split(',')
        keys = []
        keys.extend(ops)
        keys.extend(cps)
        keys.extend(lcs)
        for k in keys:
            self.RNNS[k] = nn.LSTM(config.SIZE_EXP_NODE_FEATURE, config.SIZE_EXP_NODE_FEATURE, 2)

        self.attvec = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # Att1
        self.softmaxer = nn.Softmax(dim=1)

    def forward(self, z3_exp):
        if len(z3_exp.children()) > 0:
            k = str(z3_exp.decl())
            Rnn = self.RNNS[k]
            child_feartures = torch.ones((1, config.SIZE_EXP_NODE_FEATURE))
            if torch.cuda.is_available():
                child_feartures = child_feartures.cuda()
            for chi in z3_exp.children():
                child_feartures = torch.cat((child_feartures, self.forward(chi)), 0)
            feature, _ = Rnn(child_feartures.reshape([-1, 1, config.SIZE_EXP_NODE_FEATURE]))
            return feature[-1]
        else:
            if str(z3_exp.decl()) in SymbolEmbeddings:
                return SymbolEmbeddings[str(z3_exp.decl())]
            else:
                return SymbolEmbeddings['?']

    def forward_three(self, args):
        pre_exp, trans_exp, post_exp = args
        pre_emb = self.forward(pre_exp)
        trans_emb = self.forward(trans_exp)
        post_emb = self.forward(post_exp)

        weis = torch.cat([torch.cosine_similarity(pre_emb, self.attvec),
                       torch.cosine_similarity(trans_emb, self.attvec),
                       torch.cosine_similarity(post_emb, self.attvec)], 0).reshape([1, 3])
        swis = self.softmaxer(weis)
        three_emb = torch.cat((pre_emb, trans_emb, post_emb), 0).reshape([3, config.SIZE_EXP_NODE_FEATURE])
        smt_emb = torch.mm(swis, three_emb)

        return smt_emb

    def GetParameters(self):
        res = {}
        PreFix = "Tree_LSTM_P_"
        res[PreFix + "attvec"] = self.attvec
        for ky in self.RNNS.keys():
            res.update(getParFromModule(self.RNNS[ky], prefix=PreFix + str(ky)))
        return res

    def cudalize(self):
        self.attvec = Parameter(self.attvec.cuda())
        for ky in self.RNNS.keys():
            self.RNNS[ky] = self.RNNS[ky].cuda()


# littel Test
if __name__ == "__main__":
    TR = TreeLSTM()
    from z3 import *

    x = Int('x')
    y = Int('y')

    ee1 = And(x + y < 3, x * (x % y + 2) >= Z3_abs(x * y))
    ee2 = Or(x + y < 3, x * (x % y + 2) >= Z3_abs(x * y))
    ee3 = x * (x % y + 2) >= Z3_abs(x * y)
    pp = TR.forward_three((ee1, ee2, ee3))
    print(pp)
