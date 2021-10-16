import json

import torch
from torch import nn, tensor
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.ExternalProcesses.CFG_parser import GetAllCGraphfilePath
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule
from code2inv.common.ssa_graph_builder import ProgramGraph

from code2inv.graph_encoder.embedding import EmbedMeanField
from code2inv.prog_generator.file_solver import GraphSample


class CFG_Embedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Need to prepare node type dict from the beginning.
        node_type_dict = {}
        allgpaths = GetAllCGraphfilePath()
        for gpath in allgpaths:
            graph_file = open(gpath, 'r')
            graph = ProgramGraph(json.load(graph_file))
            for node in graph.node_list:
                if not node.node_type in node_type_dict:
                    v = len(node_type_dict)
                    node_type_dict[node.node_type] = v


        self.encoder = EmbedMeanField(config.SIZE_EXP_NODE_FEATURE, len(node_type_dict), max_lv=10)
        self.attvec = Parameter(torch.randn((1,config.SIZE_EXP_NODE_FEATURE)), requires_grad=True) #Att3
        self.softmaxer = nn.Softmax(dim=1)
        self.g_list = GraphSample(cfg, [], node_type_dict)




    def forward(self, emb_smt, emb_CE, stateVec):
        self.cfg_emb = self.encoder(self.g_list)
        weighted1 = torch.mm(self.cfg_emb, stateVec.transpose(0,1)).transpose(0,1)
        cfg_emb = torch.mm(weighted1, self.cfg_emb)
        weis = torch.cat([torch.cosine_similarity(cfg_emb, self.attvec),
                       torch.cosine_similarity(emb_smt, self.attvec),
                       torch.cosine_similarity(emb_CE, self.attvec)],0).reshape([1,3])
        swis = self.softmaxer(weis)
        three_emb = torch.cat((cfg_emb, emb_smt, emb_CE), 0).reshape([3, config.SIZE_EXP_NODE_FEATURE])
        overall_feature = torch.mm(swis, three_emb)
        return overall_feature

    def GetParameters(self):
        res = {}
        PreFix = "CFG_Embedding_P_"
        res[PreFix + "attvec"] = self.attvec
        res.update(getParFromModule(self.encoder, prefix=PreFix + "encoder"))
        return res


    def cudalize(self):
        self.attvec = Parameter(self.attvec.cuda())
        self.encoder = self.encoder.cuda()