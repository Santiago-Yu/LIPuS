import argparse
import os

import torch

cmd_opt = argparse.ArgumentParser(description='Argparser')
cmd_opt.add_argument('-data_root', default=None, help='root of dataset')
cmd_opt.add_argument('-file_list', default=None, help='list of programs')
cmd_opt.add_argument('-input_graph', default=None, help='path to single input json graph')
cmd_opt.add_argument('-input_vcs', default=None, help='path to input smt2 format VCs')
cmd_opt.add_argument('-input_source_file', default=None, help='path to the source code')
cmd_opt.add_argument('-inv_grammar', default=r"code2inv/prog_generator/grammar_files/inv.grammar", help='path to invariant grammar file')
cmd_opt.add_argument('-inv_checker', default=r"code2inv.prog_generator.checkers.c_inv_checker", help='path to solver module')
cmd_opt.add_argument('-var_format', default=None, help='format of invariant variables')
cmd_opt.add_argument('-init_model_dump', default=None, help='init model dump')
cmd_opt.add_argument('-save_dir', default=None, help='root for output')
cmd_opt.add_argument('-att_dir', default=None, help='root for att output')
cmd_opt.add_argument('-log_file', default=None, help='log file')
cmd_opt.add_argument('-boogie_exe', default=None, help='boogie binary file')
cmd_opt.add_argument('-only_use_z3', default=None, type=bool, help='check everything use z3')
cmd_opt.add_argument('-aggressive_check', default=0, type=int, help='penalize verbose/unnecessary sub expression')
cmd_opt.add_argument('-ctx', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-inv_reward_type', default='any', help='any/ordered')
cmd_opt.add_argument('-phase', default='test', help='train/test')
cmd_opt.add_argument('-train_frac', default=0.9, type=float, help='fraction for training')
cmd_opt.add_argument('-tune_test', default=0, type=int, help='active search or not')

cmd_opt.add_argument('-seed', default=1, type=int, help='random seed')
cmd_opt.add_argument('-use_ce', default=0, type=int, help='whether use counter examples')
cmd_opt.add_argument('-rl_batchsize', default=1, type=int, help='batch size for rl training')
cmd_opt.add_argument('-single_sample', default=None, type=int, help='tune single program')
cmd_opt.add_argument('-replay_memsize', default=100, type=int, help='replay memsize')
cmd_opt.add_argument('-num_epochs', default=10000, type=int, help='num epochs')
cmd_opt.add_argument('-embedding_size', default=128, type=int, help='embedding size')
cmd_opt.add_argument('-s2v_level', default=10, type=int, help='# propagations of s2v')
cmd_opt.add_argument('-ce_batchsize', default=100, type=int, help='batchsize for counter example check')

cmd_opt.add_argument('-attention', default=1, type=int, help='attention for embedding')
cmd_opt.add_argument('-exit_on_find', default=0, type=int, help='exit when found')

cmd_opt.add_argument('-encoder_model', default='GNN', help='encoder model', choices=['GNN', 'LSTM', 'Param'])
cmd_opt.add_argument('-decoder_model', default='AssertAwareRNN', help='decoder model')
cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='random seed')

cmd_opt.add_argument('-op_file', default=None, type=str, help='Output File')

cmd_opt.add_argument('-use_random', default=1, type=int, help='if use randomness to choose a grammar rule')

cmd_opt.add_argument('-eps', default=0.05, type=float, help='the degree of grammar choice randomness')
cmd_opt.add_argument('-debug', default=0, type=int, help='if open debug info')
# for testing purposes- saves smt version of generated invariant
cmd_opt.add_argument('-save_smt', default=None, type=str, help='save smt version when invariant generated')
cmd_opt.add_argument('-cold_start', default=0, type=int, help='if turn on cold_start')
cmd_opt.add_argument('-unsat_core', default=0, type=int, help='if turn on usnat core counter example')
cmd_opt.add_argument('-VarEliminated', default=0, type=int, help='Only an global variable')

cmd_opt.add_argument('-checkprerep', default=1, type=int, help='Whether to check the repeating')
cmd_opt.add_argument('-const_bound', default=-1, type=int, help='set a limit to const_bound')
cmd_opt.add_argument('-octagon', default=0, type=int, help='whether to use octagon domain')
cmd_opt.add_argument('-linearcomplete', default=0, type=int, help='whether we ask the appearance of all variables')
cmd_opt.add_argument('-max_depth', default=15, type=int, help='the length of a candiatate')
cmd_opt.add_argument('-octagon_bound', default=1, type=int, help='set a limit to const_bound')
cmd_opt.add_argument('-solTime', default=100000, type=int, help='how many seconds we wait for solving the consts')
cmd_opt.add_argument('-maxLinearVarNumber', default=3, type=int, help='how many variables in one predicate we permit')
cmd_opt.add_argument('-tryLinearwhenLast', default=0, type=int, help='Open the switch to try Linear in the last time')
cmd_opt.add_argument('-self_supervised_instances', default=10, type=int, help='How many instances self supervising is '
                                                                              'using')




cmd_args, _ = cmd_opt.parse_known_args()
if torch.cuda.is_available():
    cmd_args.ctx = 'gpu'

start_time = None
import time
def tic():
    global start_time
    start_time = time.time()

def toc():
    global start_time
    cur_time = time.time()
    return cur_time - start_time 
if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)
print(cmd_args)
