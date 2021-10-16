import torch
from torch import nn, tensor

import numpy as np
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


def pca(X, k):  # k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues

    eig_val, eig_vec = np.linalg.eig(scatter_matrix)

    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True, key= lambda x:x[0])
    # select the top k eig_vec
    while len(eig_pairs) < k: # padding them
        eig_pairs.append((0, np.transpose(np.matrix([0]*len(eig_pairs[0][1])))))
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data


class CEEmbedding(nn.Module):
    def __init__(self, vars):
        super().__init__()
        self.vars = vars
        self.RNNs = {}
        for keyer in ['p', 'n', 'i_1', 'i_2']:
            self.RNNs['CE_' + keyer ] = nn.LSTM(config.SIZE_PCA_NUM, config.SIZE_EXP_NODE_FEATURE, 2)
        self.attvec = Parameter(torch.randn((1,config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
        self.softmaxer = nn.Softmax(dim=1)

    def matxlize(self, lister):
        seq = []
        for example in lister:
            seq_1 = []
            for varer in self.vars:
                if varer in example:
                    seq_1.append(int(str(example[varer])))
                else:
                    seq_1.append(0)
            seq.append(seq_1)
        if len(seq) == 0:
            seq.append([0])
        return np.matrix(seq)

    def matxlize_inductive(self, lister):
        seq_pre = []
        seq_post = []
        for example in lister:
            e_pre, e_post = example
            seq_1 = []
            seq_2 = []
            for varer in self.vars:
                if varer in e_pre:
                    seq_1.append(int(str(e_pre[varer])))
                else:
                    seq_1.append(0)

                if varer in e_post:
                    seq_2.append(int(str(e_post[varer])))
                else:
                    seq_2.append(0)
            seq_pre.append(seq_1)
            seq_post.append(seq_2)

        if len(seq_pre) == 0:
            seq_pre.append([0])
        if len(seq_post) == 0:
            seq_post.append([0])
        return np.matrix(seq_pre), np.matrix(seq_post)

    def forward(self, CE):
        matx_p = self.matxlize(CE['p'])
        matx_n = self.matxlize(CE['n'])
        matx_i1, matx_i2 = self.matxlize_inductive(CE['i'])

        pca_p = tensor(pca(matx_p, config.SIZE_PCA_NUM), dtype=torch.float32).reshape([-1,1,config.SIZE_PCA_NUM])
        pca_n = tensor(pca(matx_n, config.SIZE_PCA_NUM), dtype=torch.float32).reshape([-1,1,config.SIZE_PCA_NUM])
        pca_i1, pca_i2 = tensor(pca(matx_i1, config.SIZE_PCA_NUM), dtype=torch.float32).reshape([-1,1,config.SIZE_PCA_NUM]), \
                         tensor(pca(matx_i2, config.SIZE_PCA_NUM), dtype=torch.float32).reshape([-1,1,config.SIZE_PCA_NUM])
        if torch.cuda.is_available():
            pca_p = pca_p.cuda()
            pca_n = pca_n.cuda()
            pca_i1, pca_i2 = pca_i1.cuda(), pca_i2.cuda()
        p_emb,_ = self.RNNs['CE_p'](pca_p)
        p_emb = p_emb[-1]
        n_emb,_ = self.RNNs['CE_n'](pca_n)
        n_emb = n_emb[-1]
        i1_emb,_ = self.RNNs['CE_i_1'](pca_i1)
        i1_emb = i1_emb[-1]
        i2_emb,_ = self.RNNs['CE_i_2'](pca_i2)
        i2_emb = i2_emb[-1]

        weis = torch.cat([torch.cosine_similarity(p_emb, self.attvec),
                       torch.cosine_similarity(n_emb, self.attvec),
                       torch.cosine_similarity(i1_emb, self.attvec),
                       torch.cosine_similarity(i2_emb, self.attvec)], 0).reshape([1,4])
        swis = self.softmaxer(weis)
        three_emb = torch.cat((p_emb, n_emb, i1_emb, i2_emb), 0).reshape([4, config.SIZE_EXP_NODE_FEATURE])
        ce_emb = torch.mm(swis, three_emb)

        return ce_emb


    def GetParameters(self):
        res = {}
        PreFix = "CounterExample_P_"
        res[PreFix + "attvec"] = self.attvec
        for ky in self.RNNs.keys():
            res.update(getParFromModule(self.RNNs[ky], prefix=PreFix + str(ky)))
        return res

    def cudalize(self):
        self.attvec = Parameter(self.attvec.cuda())
        for ky in self.RNNs.keys():
            self.RNNs[ky] = self.RNNs[ky].cuda()


#unit test

if __name__ == "__main__":
    vars = ['x', 'y', 'z']
    CE = {
        'p': [{'x':1, 'y':2, 'z':3}, {'x':4, 'y':7, 'z':2}, {'x':3, 'y':4, 'z':8},
        {'x':11, 'y':4, 'z':8}],
        'n': [{'x':1, 'y':2, 'z':3}, {'x':1, 'y':2, 'z':3}, {'x':1, 'y':2, 'z':3}],
        'i': [[{'x':1, 'y':2, 'z':3}, {'x':1, 'y':2, 'z':3}], [{'x':1, 'y':2, 'z':3}, {'x':1, 'y':2, 'z':3}]]
    }
    C = CEEmbedding(vars)
    print(C(CE))