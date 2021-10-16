from pycparser import *
from pycparser import parse_file


class VarDefVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.varnames = []

    def visit_Decl(self, node):
        if 'main' in node.name:
            return
        self.varnames.append(node.name)


def get_varnames_from_source_code(path2CFile):
    try:
        astnode = parse_file(path2CFile, use_cpp=True)
    except c_parser.ParseError as e:
        return "Parse error:" + str(e)
    v = VarDefVisitor()
    v.visit(astnode)
    return v.varnames


class ConstDefVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.consts = set()

    def visit_Constant(self, node):
        if node.type == 'int':
            self.consts.add(int(node.value))


def get_consts_from_source_code(path2CFile):
    try:
        astnode = parse_file(path2CFile, use_cpp=True)
    except c_parser.ParseError as e:
        return "Parse error:" + str(e)
    v = ConstDefVisitor()
    v.visit(astnode)
    consters = list(v.consts)
    for extrac in [0, 1, -1, 2, -2, 3, -3, 6, 4]:
        if extrac not in consters:
            consters.append(extrac)
    return consters
