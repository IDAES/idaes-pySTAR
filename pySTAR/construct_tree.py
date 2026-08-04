import logging
import pprint
import re
import binarytree

# pylint: disable = logging-fstring-interpolation
LOGGER = logging.getLogger(__name__)


def add_aux_vars_to_inner_expressions(expr: str, aux_vars: dict):
    """
    Identifies inner expressions that are not univariate functions
    and introduces auxiliary variables for them.

    Parameters

    ----------
    expr : str
        Expression

    aux_vars : dict
        Dictionary to store auxiliary variables
    """
    # First, remove empty spaces if any
    modified_expression = expr.replace(" ", "")
    counter = len(aux_vars) + 1

    LOGGER.debug(
        f"\n{add_aux_vars_to_inner_expressions.__name__}: Input expr: {modified_expression}\n"
    )

    while modified_expression.find("(") >= 0:
        LOGGER.debug(
            f"{add_aux_vars_to_inner_expressions.__name__}: "
            f"Iter{counter} start expr: {modified_expression}"
        )

        last_idx = modified_expression.find(")")
        if last_idx == -1:
            raise ValueError("Mismatch in parenthesis. Check expression again!")

        open_bracket_idxs = [
            match.start()
            for match in re.finditer(r"\(", modified_expression[:last_idx])
        ]
        start_idx = open_bracket_idxs[-1]

        if start_idx == 0 or modified_expression[start_idx - 1] in "(+-*/":
            aux_var_name = f"aux_var_{counter}"
            aux_vars[aux_var_name] = modified_expression[start_idx + 1 : last_idx]
            modified_expression = (
                modified_expression[:start_idx]
                + aux_var_name
                + modified_expression[last_idx + 1 :]
            )

        else:
            # This is likely a univariate function
            aux_var_name = f"uni_var_{counter}"
            op_start_idx = max(
                modified_expression[:start_idx].rfind(op) for op in "(+-*/"
            )
            aux_vars[aux_var_name] = modified_expression[
                op_start_idx + 1 : last_idx + 1
            ]
            modified_expression = (
                modified_expression[: op_start_idx + 1]
                + aux_var_name
                + modified_expression[last_idx + 1 :]
            )

        LOGGER.debug(
            f"{add_aux_vars_to_inner_expressions.__name__}: "
            f"Iter{counter} end expr: {modified_expression}"
        )
        counter += 1

    LOGGER.debug(
        f"\n{add_aux_vars_to_inner_expressions.__name__} Output expr: {modified_expression}\n"
    )

    return modified_expression


def add_aux_var_for_negative_var(expr: str, aux_vars: dict):
    """
    If the expression begins with a negative sign, this function
    defines an auxiliary variable to remove

    Parameters
    ----------
    expr : str
        Expression

    aux_vars : dict
        Dictionary to store auxiliary variables
    """
    if expr[0] != "-":
        return expr

    counter = len(aux_vars) + 1
    aux_var_name = f"neg_var_{counter}"

    if any(op in expr[1:] for op in "+*-/"):
        modified_expression = expr[1:]
        op_idxs = [
            modified_expression.find(op) if modified_expression.find(op) > 0 else 10000
            for op in "+*-/"
        ]
        first_op_idx = min(op_idxs) + 1
        return aux_var_name + expr[first_op_idx:]

    return aux_var_name


def add_aux_var_for_power_function(expr: str, aux_vars: dict):
    """
    Adds an auxiliary variable for power functions

    Parameters
    ----------
    expr : str
        Expression

    aux_vars : dict
        Dictionary to store auxiliary variables
    """
    # The function can handle both ^ and ** operators for powers
    modified_expression = expr.replace("^", "**")
    counter = len(aux_vars) + 1

    # Brackets are not handled correctly if the input is a univariate
    # function. So, we strip brackets, then introduce auxiliary variables
    # and then add them back at the end
    if modified_expression.find("(") > 0:
        modified_expression = re.findall(r"\((.*?)\)", modified_expression)[0]

    while modified_expression.find("**") > 0:
        power_idx = modified_expression.find("**")
        prev_op_idx = max(modified_expression[:power_idx].rfind(op) for op in "+-*/")
        op_idxs = [modified_expression[power_idx + 2 :].find(op) for op in "*+-/"]
        if max(op_idxs) == -1:
            next_op_idx = -1
        else:
            next_op_idx = power_idx + 2 + min(i if i >= 0 else 100000 for i in op_idxs)

        exponent = (
            modified_expression[power_idx + 2 :]
            if next_op_idx == -1
            else modified_expression[power_idx + 2 : next_op_idx]
        )
        base = (
            modified_expression[:power_idx]
            if prev_op_idx == -1
            else modified_expression[prev_op_idx + 1 : power_idx]
        )

        aux_var_name = f"pow_var_{counter}"
        aux_expr = base + "**" + exponent
        aux_vars[aux_var_name] = aux_expr
        modified_expression = modified_expression.replace(aux_expr, aux_var_name)
        counter += 1

    # Add the univariate function and parenthesis back (if they were removed before)
    if expr.find("(") > 0:
        modified_expression = expr[: expr.find("(") + 1] + modified_expression + ")"

    return modified_expression


def get_sub_tree(expr: str):
    """
    Returns a binary tree for a given subtree

    Parameters
    ----------
    expr : str
        (Sub)expression (should not contain spaces)

    Returns
    -------
    binarytree.Node
        Binary tree

    Raises
    ------
    ValueError
        If tree construction fails for a given subtree
    """
    op_map = {"-": "diff", "+": "sum", "*": "mult", "/": "div"}
    bin_ops = list(op_map)

    if all(op not in expr for op in bin_ops):
        return binarytree.Node(value=expr)

    for op in bin_ops:
        if op in expr:
            child_exprs = expr.split(op, maxsplit=1)
            return binarytree.Node(
                value=op_map[op],
                left=get_sub_tree(child_exprs[0]),
                right=get_sub_tree(child_exprs[1]),
            )

    raise ValueError(f"Cannot handle this expression {expr}")


def get_expression_tree(expr: str):
    """
    Constructs a binary expression tree for a given expression

    Parameters
    ----------
    expr : str
        Expression (Only use parenthesis. Do not use square or curly brackets)
    """
    aux_vars = {}  # Contains auxiliary variable names and expressions
    aux_var_trees = {}  # Contains auxiliary variable names and binary trees

    # First, replace auxiliary variables to simplify inner expressions
    modified_expr = add_aux_vars_to_inner_expressions(expr, aux_vars)

    # Replace power functions with auxiliary variables
    modified_expr = add_aux_var_for_power_function(modified_expr, aux_vars)

    for a_var, a_expr in aux_vars.copy().items():
        aux_vars[a_var] = add_aux_var_for_power_function(a_expr, aux_vars)

    # Next, modify the expression if it starts with a negative sign
    modified_expr = add_aux_var_for_negative_var(modified_expr, aux_vars)

    for a_var, a_expr in aux_vars.copy().items():
        aux_vars[a_var] = add_aux_var_for_negative_var(a_expr, aux_vars)

    LOGGER.debug(f"\nModified Expression: {modified_expr}")
    LOGGER.debug(
        f"List of auxiliary variables:\n {pprint.pformat(aux_vars, indent=4)}\n"
    )

    # Construct trees for auxiliary variables
    for a_var, a_expr in aux_vars.items():
        if "aux_var" in a_var or a_var == "neg_var":
            # This is a simple expression
            aux_var_trees[a_var] = get_sub_tree(a_expr)

        if "uni_var" in a_var:
            # This is a univariate function
            op = a_expr[: a_expr.find("(")]
            argument = get_sub_tree(re.findall(r"\((.*?)\)", a_expr)[0])
            aux_var_trees[a_var] = binarytree.Node(value=op, right=argument)

        if "pow_var" in a_var:
            base_exponent = a_expr.split("**")
            aux_var_trees[a_var] = binarytree.Node(
                value="**",
                left=get_sub_tree(base_exponent[0]),
                right=get_sub_tree(base_exponent[1]),
            )

    # Construct the expression tree
    tree = get_sub_tree(modified_expr)
    if tree.value in aux_vars:
        tree = aux_var_trees[tree.value]

    # Substitute sub trees for auxiliary variables
    is_tree_modified = True
    while is_tree_modified:
        nodes_list = tree.levelorder
        for node in nodes_list:
            if node.left is not None and node.left.value in aux_vars:
                node.left = aux_var_trees[node.left.value]

            if node.right is not None and node.right.value in aux_vars:
                node.right = aux_var_trees[node.right.value]

        is_tree_modified = len(tree.levelorder) != len(nodes_list)

    return tree
