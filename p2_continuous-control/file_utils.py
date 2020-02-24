# -*- encoding: utf-8 -*-
"""
@File           :   file_utils.py
@Time           :   2020_01_28-23:02:59
@Author         :   zhenwang
@Description    :
  - Version 1.0.0: File created.
"""


def get_vars_from_file(mod_path, default=None, raise_exception=False):
    import ast
    ModuleType = type(ast)
    with open(mod_path, "r") as file_mod:
        data = file_mod.read()

    try:
        ast_data = ast.parse(data, filename=mod_path)
    except:
        if raise_exception:
            raise
        print("Syntax error 'ast.parse' can't read %r" % mod_path)
        import traceback
        traceback.print_exc()

    return_value = {}

    if ast_data:
        for body in ast_data.body:
            if body.__class__ == ast.Assign:
                if len(body.targets) == 1:
                    try:
                        return_value[body.targets[0].id] = ast.literal_eval(body.value)
                    except:
                        if raise_exception:
                            raise
                        print("AST error parsing for %r" % (mod_path))
                        import traceback
                        traceback.print_exc()
    return return_value if return_value else default


# example use
# a = 0
# variables = get_vars_from_file(__file__)
# print(variables)
