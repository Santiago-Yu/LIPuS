from __future__ import print_function

import io
import os
import sys
import tokenize

import numpy as np
import torch
import json
import random
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain

from code2inv.coldstart.cold_starter import expHeristicGenerater
from code2inv.common.ssa_graph_builder import ProgramGraph, GraphNode
from code2inv.common.constants import *
from code2inv.common.cmd_args import cmd_args, tic, toc
from code2inv.common.checker import stat_counter, boogie_result

from code2inv.graph_encoder.embedding import EmbedMeanField, LSTMEmbed, ParamEmbed
from code2inv.prog_generator.rl_helper import rollout, actor_critic_loss
from code2inv.prog_generator.tree_decoder import GeneralDecoder

from code2inv.graph_encoder.s2v_lib import S2VLIB, S2VGraph
from code2inv.templeter.const_smt_helper import findconsters
from code2inv.prog_generator.checkers.c_inv_checker import infix_postfix, postfix_prefix, \
    stringify_prefix_stack

class GraphSample(S2VGraph):
    def __init__(self, pg, vc_list, node_type_dict):
        super(GraphSample, self).__init__(pg, node_type_dict)
        self.sample_index = 0
        self.db = None
        self.vc_list = vc_list

def condense_minusF(inv_tokens):
    op_list = ["+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!=", "and", "or"]
    un_op_list = ["+", "-"]
    old_list = list(inv_tokens)
    new_list = list(inv_tokens)
    while True:
        for idx in range(len(old_list)):
            if old_list[idx] in un_op_list:
                if idx == 0 or old_list[idx-1] in op_list or old_list[idx-1] == "(":
                    new_list[idx] = '(' + old_list[idx] + ' ' +old_list[idx+1] + ')'
                    new_list[idx+1:] = old_list[idx+2:]
                    break
        if old_list == new_list:
            break
        else:
            old_list = list(new_list)
    return new_list


def convertexp2smt_minusF(inv):
    inv = inv.replace("&&", "and", -1)
    inv = inv.replace("||", "or", -1)
    b = io.StringIO(inv)
    t = tokenize.generate_tokens(b.readline)
    inv_tokenized = []
    for a in t:
        if a.string != "":
            inv_tokenized.append(a.string)
    inv = stringify_prefix_stack(postfix_prefix(infix_postfix(condense_minusF(inv_tokenized))))
    inv = inv.replace("==", "=", -1)
    return inv

def vol2lat_getSize(root, g):
    #1.transform root into an smt and write it into a file.
    #print(root)
    varnamesset = g.pg.core_vars
    smt_exp= convertexp2smt_minusF(str(root))
    #print(smt_exp)
    presmt = "(set-logic QF_LIA)\n"
    aftersmt = "(check-sat)\n(exit)\n"
    smtstr = ""
    smtstr += presmt
    for varname in varnamesset:
        smttmp = '(declare-const ' + varname + ' Int) '
        smtstr += smttmp
    smtstr += "(assert " + smt_exp + ')\n'
    smtstr += aftersmt
    if cmd_args.debug:
        print(smtstr)
    tmpFilename = "tmp_vol2lat.smt2"
    filer = open(tmpFilename, "w")
    filer.write(smtstr)
    filer.close()
    #2.run the vol2lat program
    CurDir = "/mnt/g/reproduce/code2inv-ice-templete-master/code2inv/prog_generator"
    programLocation = "/mnt/g/reproduce/VOL2LAT-master/vol2lat"
    resultfile = "tmpresult.txt"
    if os.path.exists(resultfile):
        os.remove(resultfile)
    cmd_str = "wsl -e " + programLocation + " -L -w=32 " + CurDir + "/" +  tmpFilename + " " + CurDir + "/" + resultfile
    cmd_str += " >waste.txt"
    if cmd_args.debug:
        print(cmd_str)
    os.chdir("G:/reproduce/VOL2LAT-master/")
    os.system(cmd_str)
    #3.get the answer from the output and return.
    os.chdir("G:/reproduce/code2inv-ice-templete-master/code2inv/prog_generator")
    spacesize = float(open(resultfile, "r").read().split(' ')[0])
    if cmd_args.debug:
        print(spacesize)
    return spacesize


def Main(cmd_args):
    if cmd_args.debug:
        recordfile = open("records_leastSapce.txt", "w")
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    tic()
    params = []

    graph = None
    node_type_dict = {}
    vc_list = []

    with open(cmd_args.input_graph, 'r') as graph_file:
        graph = ProgramGraph(json.load(graph_file))
        for node in graph.node_list:
            if not node.node_type in node_type_dict:
                v = len(node_type_dict)
                node_type_dict[node.node_type] = v

    if graph is not None:
        if cmd_args.encoder_model == 'GNN':
            encoder = EmbedMeanField(cmd_args.embedding_size, len(node_type_dict), max_lv=cmd_args.s2v_level)
        elif cmd_args.encoder_model == 'LSTM':
            encoder = LSTMEmbed(cmd_args.embedding_size, len(node_type_dict))
        elif cmd_args.encoder_model == 'Param':
            g_list = GraphSample(graph, vc_list, node_type_dict)
            encoder = ParamEmbed(cmd_args.embedding_size, g_list.pg.num_nodes())
        else:
            raise NotImplementedError

        coldstart_templetes = []
        if cmd_args.cold_start:
            if os.path.exists(cmd_args.input_source_file):
                coldstart_templetes = expHeristicGenerater(cmd_args.input_source_file)


        decoder = GeneralDecoder(cmd_args.embedding_size, GraphSample(graph, vc_list, node_type_dict), coldstart_templetes)

        if cmd_args.init_model_dump is not None:
            encoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.encoder'))
            decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder'))

        params.append(encoder.parameters())
        params.append(decoder.parameters())

        optimizer = optim.Adam(chain.from_iterable(params), lr=cmd_args.learning_rate)
        counter_mine = 0
        if cmd_args.debug:
            recordfile.write(str(float(2**(32*len(GraphSample(graph, vc_list, node_type_dict).pg.core_vars))))+"\n")
            recordfile.flush()

        for epoch in range(cmd_args.num_epochs):
            best_reward = -5.0
            best_root = None
            tested_roots = []

            acc_reward = 0.0
            pbar = tqdm(range(100), file=sys.stdout)
            for k in pbar:

                g_list = GraphSample(graph, vc_list, node_type_dict)
                node_embedding_batch = encoder(g_list)

                total_loss = 0.0
                embedding_offset = 0

                for b in range(cmd_args.rl_batchsize):
                    g = GraphSample(graph, vc_list, node_type_dict)
                    node_embedding = node_embedding_batch
                    # Here we would like to take cmd_args.self_supervised_instances answers and compare them. changing the rewards.
                    Selfs = []
                    least_space = 0
                    max_space = float(2**(32*len(g.pg.core_vars)))

                    least_space_wehave = float("inf")
                    for i in range(cmd_args.self_supervised_instances):
                        nll_list, value_list, reward_list, root, _ = rollout(g, node_embedding, decoder,
                                                                             use_random=cmd_args.use_random,
                                                                             eps=cmd_args.eps)
                        counter_mine += 1
                        if reward_list[-1] > 0:
                            space_size = vol2lat_getSize(root, g)
                            # if space_size > max_space:
                            #     max_space = space_size
                            if space_size < least_space_wehave:
                                least_space_wehave = space_size
                        else:
                            space_size = -1 # that means the answer is not available.
                        if cmd_args.debug:
                            print("root ", root, "; space size ", space_size)
                        Selfs.append([nll_list, value_list, reward_list, root, space_size])
                    if (max_space - least_space_wehave) > 0.1*max_space:
                        if cmd_args.debug:
                            print("The least space size is", least_space_wehave)
                            recordfile.write(str(least_space_wehave) + "    " + str(counter_mine) +"\n")
                            recordfile.flush()
                    for trail in Selfs:
                        if trail[-1] != -1:
                            if (max_space - trail[-1]) > 0:
                                trail[-3][-1] *= (max_space - trail[-1]) / (max_space - least_space) * 10 # rearrange the size
                            else:
                                trail[-3][-1] *= 0.1
                        nll_list, value_list, reward_list, root = trail[0:-1]

                        tested_roots.append(root)
                        if reward_list[-1] > best_reward:
                            best_reward = reward_list[-1]
                            best_root = root

                        acc_reward += np.sum(reward_list) / cmd_args.rl_batchsize
                        loss = actor_critic_loss(nll_list, value_list, reward_list)
                        total_loss += loss

                optimizer.zero_grad()
                loss = total_loss / cmd_args.rl_batchsize
                pbar.set_description('avg reward: %.4f || loss: %.4f' % (acc_reward / (k + 1), loss))
                loss.backward()
                optimizer.step()

            g = GraphSample(graph, vc_list, node_type_dict)
            node_embedding = encoder(g)

            while True:
                _, _, _, root, trivial = rollout(g, node_embedding, decoder, use_random=cmd_args.use_random,
                                                 eps=0.0)  # eps =0.0 means not use random??
                if trivial == False:
                    break

            print('epoch: %d, average reward: %.4f, Random: %s, result_r: %.4f' % (
            epoch, acc_reward / 100.0, root, boogie_result(g, root)))
            print("best_reward:", best_reward, ", best_root:", best_root)
            stat_counter.report_global()
            if cmd_args.save_dir is not None:
                torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-%d.encoder' % epoch)
                torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-%d.decoder' % epoch)
                torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-latest.encoder')
                torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-latest.decoder')


if __name__ == '__main__':
    Main(cmd_args)