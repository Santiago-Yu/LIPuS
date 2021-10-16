import json
import os

from code2inv.common.ssa_graph_builder import ProgramGraph


def parseCFG(path2CFG):
    graph = ProgramGraph(json.load(open(path2CFG,"r")))
    return graph


def GetAllCGraphfilePath():
    filepaths = []
    dir_linear = r"Benchmarks/Linear/c_graph/"
    dir_nonlinear = r"Benchmarks/NL/c_graph/"
    files = os.listdir(dir_linear)
    for f in files:
        filepaths.append(dir_linear + f)
    files = os.listdir(dir_nonlinear)
    for f in files:
        filepaths.append(dir_nonlinear + f)
    return filepaths