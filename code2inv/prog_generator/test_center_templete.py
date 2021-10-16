from __future__ import print_function

import os
import sys
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

from code2inv.coldstart.cold_starter import expHeristicGenerater, parseIntoAction

from code2inv.common.ssa_graph_builder import ProgramGraph, GraphNode
from code2inv.common.constants import *
from code2inv.common.cmd_args import cmd_args, tic, toc
from code2inv.common.checker import stat_counter, boogie_result

from code2inv.graph_encoder.embedding import EmbedMeanField, LSTMEmbed, ParamEmbed
from code2inv.prog_generator.file_solver import GraphSample
from code2inv.prog_generator.rl_helper import rollout, actor_critic_loss
from code2inv.prog_generator.tree_decoder import GeneralDecoder

from code2inv.graph_encoder.s2v_lib import S2VLIB, S2VGraph

# This file is used to evaluate the performance of Code2inv
from code2inv.common.cmd_args import cmd_args
from code2inv.common.checker import stat_counter, code_ce_dict
from code2inv.templeter.const_smt_helper import VarEliminated


def Main(node_type_dict, cmd_args, encoder_arg=None, decoder_arg=None, timeLimit=600.0):
    code_ce_dict.clear()
    stat_counter.clear()
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    tic()
    isOriginalOctagon = None

    graph = None

    vc_list = []
    #
    with open(cmd_args.input_graph, 'r') as graph_file:
        graph = ProgramGraph(json.load(graph_file))
        # for node in graph.node_list:
        #     if not node.node_type in node_type_dict:
        #         v = len(node_type_dict)
        #         node_type_dict[node.node_type] = v



    params = []
    if graph is not None:
        if encoder_arg is not None and cmd_args.encoder_model != 'Param':
            encoder = encoder_arg
        elif cmd_args.encoder_model == 'GNN':
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

        if decoder_arg is not None: #reopen the cold start
            decoder = decoder_arg
            if cmd_args.cold_start:
                decoder.cold_start_queue = \
                    parseIntoAction(coldstart_templetes, GraphSample(graph, vc_list, node_type_dict))
                decoder.cold_start = cmd_args.cold_start
                decoder.cold_start_tried = 0
            else:
                decoder.cold_start = cmd_args.cold_start
                decoder.cold_start_tried = 0
        else:
            decoder = GeneralDecoder(cmd_args.embedding_size, GraphSample(graph, vc_list, node_type_dict),
                                 coldstart_templetes)

        if cmd_args.init_model_dump is not None and decoder_arg is None:
            encoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.encoder'))
            decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder'))

        params.append(encoder.parameters())
        params.append(decoder.parameters())

        optimizer = optim.Adam(chain.from_iterable(params), lr=cmd_args.learning_rate)

        found = False

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
                    nll_list, value_list, reward_list, root, _ = rollout(g, node_embedding, decoder,
                                                                         use_random=cmd_args.use_random,
                                                                         eps=cmd_args.eps)
                    if reward_list[-1] > 0 : # we can return now
                        if decoder_arg is None:
                            if isOriginalOctagon is not None:
                                cmd_args.octagon = isOriginalOctagon
                            return True, str(root), toc(), stat_counter.stats_dict[g.sample_index], encoder, decoder
                        else:
                            found =True
                            found_root = root
                            found_g = g


                    elif toc() > timeLimit: # we say 10 minutes is a limit.
                        if isOriginalOctagon is not None:
                            cmd_args.octagon = isOriginalOctagon
                        return False, None, toc(), stat_counter.stats_dict[g.sample_index], encoder, decoder

                    elif toc() > timeLimit * 2 / 3 and cmd_args.tryLinearwhenLast and cmd_args.octagon == 1:# Try Linear in the rest of 1/3 time
                        print("Open Linear!")

                        isOriginalOctagon = cmd_args.octagon
                        cmd_args.octagon = 0
                        # we also have to clear the counter examples.
                        code_ce_dict.clear()

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
                if found:  # Delayed return.
                    if isOriginalOctagon is not None:
                        cmd_args.octagon = isOriginalOctagon
                    return True, str(found_root), toc(), stat_counter.stats_dict[found_g.sample_index], encoder, decoder



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





def Solving(cmd_args, timeLimit=600.0):
    # iterate each file
    repeater = 1
    timeLimit = timeLimit/ repeater
    log_path = "testResults/testOn+Param+templete_smt_solving+debug+100ce+15deepth+Nolinearcomplete+4MAXvar+octagon_linearInthelast.txt"
    filer = open(log_path,"w")
    filer.write(str(cmd_args))
    filer.write("\n#################################################################################\n")


    prePath_g = r"../../benchmarks/C_instances/c_graph/"
    prePath_s = r"../../benchmarks/C_instances/c_smt2/"
    prePath_c = r"../../benchmarks/C_instances/c/"


    solved_in_total = 0
    avetime_sol = 0.0
    most_time_sol = 0.0
    min_time_sol = timeLimit
    encoder_arg, decoder_arg = None, None
    problems = range(1, 134)
    #problems = [93]
    # Need to prepare node type dict from the beginning.
    node_type_dict = {}
    for i in problems:
        gfilename = str(i) + ".c.json"
        path_g = prePath_g + gfilename
        graph_file = open(path_g, 'r')
        graph = ProgramGraph(json.load(graph_file))
        for node in graph.node_list:
            if not node.node_type in node_type_dict:
                v = len(node_type_dict)
                node_type_dict[node.node_type] = v




    while repeater > 0:
        problems_next = []
        for i in problems:
            if i in [26, 27, 31, 32, 61, 62, 72, 75, 106]: # not solvable
                continue
            print("Solving problem:    ", i)
            gfilename = str(i) + ".c.json"
            path_g = prePath_g + gfilename
            sfilename = str(i) + ".c.smt"
            path_s = prePath_s + sfilename
            cfilename = str(i) + ".c"
            path_c = prePath_c + cfilename

            cmd_args.input_graph = path_g
            cmd_args.input_vcs = path_s
            cmd_args.input_source_file = path_c

            solved,solution,timeUsed, counterdict, encoder_arg, decoder_arg = Main(node_type_dict, cmd_args,
                                                                                   encoder_arg=None,
                                                                                   decoder_arg=None,
                                                                                   timeLimit=timeLimit)
            if cmd_args.VarEliminated:
                print("used VarEliminated!")
                cmd_args.VarEliminated = False
            if solved:
                solved_in_total+=1
                avetime_sol += timeUsed
                if most_time_sol < timeUsed:
                    most_time_sol = timeUsed
                if min_time_sol > timeUsed:
                    min_time_sol = timeUsed
            else:
                problems_next.append(i)

            record = "Solving " + gfilename + ":::\n"
            record += "solved:" + str(solved) + "   timeUsd:" + str(timeUsed) + "   solution:" + str(solution) + "\n"
            record += str(counterdict) + "\n"
            filer.write(record)
        cmd_args.cold_start = False
        problems = problems_next
        repeater -= 1
    filer.write("\n##################################################\n")
    filer.write("Summary:\n")
    conclusion = "solved_in_total: " + str(solved_in_total) + " avetime_sol: " + str(avetime_sol) + \
                 " most_time_sol: " + str(most_time_sol) + " min_time_sol: " + str(min_time_sol) + "\n"
    filer.write(conclusion)
    filer.close()













if __name__ == "__main__":
    Solving(cmd_args)