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
from tqdm import tqdm
import importlib

from code2inv.common.ssa_graph_builder import ProgramGraph, GraphNode, ExprNode
from code2inv.common.constants import *
from code2inv.common.cmd_args import cmd_args
from code2inv.common.checker import boogie_result, z3_precheck, z3_precheck_expensive, stat_counter
from code2inv.prog_generator.tree_decoder import genExprTree, GeneralDecoder, InvariantTreeNode, fully_expanded_node
checker_module = importlib.import_module(cmd_args.inv_checker)

class RLEnv(object):
    def __init__(self, s2v_graph, decoder):
        self.s2v_graph = s2v_graph
        self.pg = s2v_graph.pg
        self.decoder = decoder
        self.reset()

    def reset(self):
        try:
            self.root = None
            self.inv_candidate = None
            self.terminal = False
            self.trivial_subexpr = False
            self.expr_in_and = set()
            self.expr_in_or = set()
            self.used_core_vars = set()
        except RecursionError:
            print("ERROR- Non Terminating grammar found")
            exit(-1)

    def num_vars(self):
        return len(self.pg.raw_variable_nodes)

    def num_consts(self):
        return len(self.pg.const_nodes)

    def pg_nodes(self):
        return self.pg.node_list

    def update_used_core_var(self, var_node):
        if var_node.name in self.pg.core_vars:
            self.used_core_vars.add(var_node.name)

    def constraint_satisfied(self):
        return len(self.used_core_vars) == len(self.pg.core_vars)

    def insert_subexpr(self, node, subexpr_node, eps = cmd_args.eps, upflag=False):
        if node is None:
            node = genExprTree(RULESET, "S")
            return self.insert_subexpr(node, subexpr_node, upflag=upflag)
        elif node.name == "" and node.rule == "p" and node.expanded is None:
            if upflag:
                return node, False
            node = subexpr_node
            node.expanded = True
            return node, True
        elif node.name == "" and node.rule == "p" and node.expanded == True:
            return node, False
        elif node.rule is not None and len(node.children) == 0:
            w = nn.Linear(cmd_args.embedding_size, len(RULESET[node.rule]))
            logits = w(self.decoder.latent_state)
            
            ll = F.log_softmax(logits, dim=1)
            if self.decoder.cold_start is not None and self.decoder.cold_start:
                picked = torch.tensor(self.decoder.cold_start_queue.popleft())
            elif self.use_random:
                scores = torch.exp(ll) * (1 - eps) + eps / ll.shape[1]
                picked = torch.multinomial(scores, 1)
            else:            
                _, picked = torch.max(ll, 1)
            picked = picked.view(-1)        

            node = genExprTree(RULESET, node.rule, picked)
            return self.insert_subexpr(node, subexpr_node, upflag=upflag)

        elif len(node.children) > 0:
            last_junct = ""
            #upflag = False
            for i in range(len(node.children)):
                node.children[i], node_update = self.insert_subexpr(node.children[i], subexpr_node, upflag=upflag)
                if node_update:
                    return node, node_update
                #     upflag = True
            if upflag:
                return node, upflag
            return node, False
        else:
            return node, False


    def step(self, subexpr_node, node_embedding, use_random, eps):
        self.use_random = use_random
        reward = 0.0        
        self.inv_candidate, updated = self.insert_subexpr(self.inv_candidate, subexpr_node)
        self.root = self.inv_candidate.clone_expanded()

        self.terminal = fully_expanded_node(self.inv_candidate)
        self.root.state = None
        if self.inv_candidate.check_rep_pred() and cmd_args.checkprerep:
            if cmd_args.debug:
                print("check_rep_pred failed!:  ", str(self.root))
            self.trivial_subexpr = True
        else:
            try:
                if checker_module.is_trivial(cmd_args.input_vcs, str(subexpr_node)):
                    self.trivial_subexpr = True
                else:
                    reward += -0.05  # Is it too much to give this a 0.5? what will happen if we give it a minor score?
                    # we give it a minus to punish long expr
            except Exception as e:
                reward += -0.05

        if self.trivial_subexpr:
            #Debug
            if cmd_args.debug and self.decoder.cold_start:
                print("trivial cold start", self.root)
            reward += -2.0
            if not self.decoder.cold_start: # dot not terminal a subexpr when cold starting
                self.terminal = True

        if self.terminal:
            if cmd_args.debug:
                if self.decoder.cold_start:
                    self.decoder.cold_start_tried += 1



            if not self.trivial_subexpr or self.decoder.cold_start: #self.constraint_satisfied():
                try:
                    r = boogie_result(self.s2v_graph, self.root)
                    reward += r
                except Exception as e:
                    print(e)
                    raise e
                    if str(e) == "Not implemented yet":
                        raise e
                    if cmd_args.debug:
                        print(e)
                    reward += -6.0
            else:
                reward += -4.0


            if cmd_args.cold_start and len(self.decoder.cold_start_queue) == 0 and self.decoder.cold_start: # If we finished all cold starts, we turn off the cold_start
                self.decoder.cold_start = False
                if cmd_args.debug:
                    print("Cold Start Over!!!!")
                    print("tried in total:  ", self.decoder.cold_start_tried)

        return reward, self.trivial_subexpr

    def available_var_indices(self, list_vars):
        list_indices = []
        if list_vars and len(list_vars) > 0:
            for var in self.pg.raw_variable_nodes:
                if var in list_vars:
                    list_indices.append(self.pg.raw_variable_nodes[var].index)
            list_indices.sort()

            if len(list_indices) == 0:
                list_indices = list(range(len(self.pg.raw_variable_nodes)))

            return list_indices
        else:
            return list(range(len(self.pg.raw_variable_nodes)))

    def is_finished(self):
        return self.terminal

def rollout(g, node_embedding, decoder, use_random, eps):
    nll_list = []
    value_list = []
    reward_list = []
    trivial = False
    env = RLEnv(g, decoder)
    while not env.is_finished():
        try:
            and_or, subexpr_node, nll, vs, latent_state = decoder(env, node_embedding, use_random=use_random, eps = eps)

            reward, trivial = env.step(subexpr_node, node_embedding, use_random, eps)

            nll_list.append(nll)
            value_list.append(vs)
            
            root = env.root
            reward_list.append(reward)
        except Exception as e:
            if cmd_args.debug:
                print("EXCEPTION", e)
                raise
            nll_list.append(decoder.nll)
            value_list.append(decoder.est)
            reward_list.append(-6.0)
            pass

    if not env.trivial_subexpr:
        if cmd_args.decoder_model == 'AssertAware':            
            assert env.constraint_satisfied()

    return nll_list, value_list, reward_list, root, trivial

def actor_critic_loss(nll_list, value_list, reward_list):
    r = 0.0
    rewards = []
    for t in range(len(reward_list) - 1, -1, -1):
        r = r + reward_list[t] # accumulated future reward
        rewards.insert(0, r / 10.0)
    # DEBUG
    if cmd_args.debug == True:
        print("!!!!!!!rewards:", rewards)
    # DEBUG
    policy_loss = 0.0

    targets = []
    for t in range(len(reward_list)):        
        difference = rewards[t] - value_list[t].data[0, 0]
        policy_loss += nll_list[t] * (difference)
        targets.append(Variable(torch.Tensor([ [rewards[t]] ])))

    policy_loss /= len(reward_list)

    value_pred = torch.cat(value_list, dim=0)
    targets = torch.cat(targets, dim=0)    
    value_loss = F.mse_loss(value_pred, targets)
    #YSW
    #value_muster = -value_pred.sum()
    #END

    loss = policy_loss + value_loss #+ value_muster

    return loss
