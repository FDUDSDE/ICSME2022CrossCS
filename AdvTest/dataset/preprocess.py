import json
from copy import deepcopy
import random
import copy
import ast
import os


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.variables = []
        self.function_name = ''

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.append(target.id)
            elif isinstance(target, ast.Tuple):
                for element in target.elts:
                    if isinstance(element, ast.Name):
                        self.variables.append(element.id)
        self.generic_visit(node)

    def visit_arguments(self, node):
        for arg in node.args:
            self.variables.append(arg.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.function_name = node.name
        self.generic_visit(node)

    def report(self):
        return [self.variables, self.function_name]


def format_str(string, variables, function_name):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    if string in variables and string != 'self':
        string = 'var' + str(variables.index(string))
    if string == function_name:
        string = 'func'
    return string


def variable_function_rename(code, tokens):
    try:
        tree = ast.parse(code)
        analyzer = Analyzer()
        analyzer.visit(tree)
        [variables, function_name] = analyzer.report()
        codestring_list = [format_str(token, variables, function_name) for token in tokens]
        return codestring_list
    except Exception as e:
        print(e)
        return ''


cnt = 0
with open("train_temporary.jsonl", 'w') as f, open('train.jsonl', 'r') as f1:
    for line in f1:
        line = json.loads(line)
        if cnt % 100 == 0:
            print(cnt)
        if "code_tokens" in line:
            code_aug_tokens = variable_function_rename(line["code"], line["code_tokens"])
            code_tokens = line["code_tokens"]
            line_aug = deepcopy(line)
            line_aug["code_tokens"] = code_aug_tokens
            line_aug["code_aug_tokens"] = code_tokens
            line_aug["idx"] = cnt
            line["code_aug_tokens"] = code_aug_tokens
            f.write(json.dumps(line_aug)+'\n')
            cnt += 1
            line["idx"] = cnt
            f.write(json.dumps(line)+'\n')
            cnt += 1


with open('train_temporary.jsonl', 'r') as f:
    data = f.readlines()
    data_shuffle = copy.copy(data)

random.seed(42)
random.shuffle(data_shuffle)

with open('train_aug.jsonl', 'w') as f1:
    for line in data_shuffle:
        f1.write(line)

os.remove('train_temporary.jsonl')
