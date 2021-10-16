import argparse

from code2inv.common.cmd_args import cmd_opt


def parseArgs():
    cmd_opt.add_argument('-g', '--graph', dest='Graph', type=str, required=True, help='The path to the CFG')
    cmd_opt.add_argument('-s', '--smt', dest='SMT', type=str, required=True, help='The path to the SMT file')
    cmd_opt.add_argument('-c', '--cfile', dest='Cfile', type=str, required=True, help='The path to the source file')
    result = cmd_opt.parse_args()
    return result.Cfile, result.Graph, result.SMT
