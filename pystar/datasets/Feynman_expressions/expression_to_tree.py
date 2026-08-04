# pip install sympy graphviz
# Also install Graphviz system package:
# - macOS: brew install graphviz
# - Ubuntu/Debian: sudo apt-get install graphviz
# - Windows: choco install graphviz  (or download installer) and ensure "dot" is on PATH

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from graphviz import Digraph
import sympy as sp


def sympy_to_graphviz(expr, out_name="expr_tree", fmt="png"):
    dot = Digraph(comment="Expression Tree", format=fmt)
    counter = {"i": 0}

    # Map SymPy operation names to mathematical symbols
    op_symbols = {
        "Add": "+",
        "Mul": "*",
        "Pow": "^",
        "Sub": "-",
        "Div": "/",
    }

    def new_id():
        counter["i"] += 1
        return f"n{counter['i']}"

    def label(node):
        # Operator/function nodes vs leaf nodes
        if isinstance(node, sp.Symbol):
            return str(node)
        if isinstance(node, sp.Number):
            return str(node)
        # For things like Add/Mul/Pow/Sin/etc - use symbols if available
        func_name = node.func.__name__
        return op_symbols.get(func_name, func_name)

    def walk(node):
        nid = new_id()
        dot.node(nid, label(node))

        # SymPy stores children in .args
        for child in getattr(node, "args", []):
            cid = walk(child)
            dot.edge(nid, cid)

        return nid

    walk(expr)

    # Try to render, but if Graphviz executable is not installed, save DOT file instead
    try:
        dot.render(out_name, cleanup=True)  # writes out_name.(png/svg/...)
        return f"{out_name}.{fmt}"
    except Exception as e:
        # If rendering fails (e.g., Graphviz not installed), save the .dot file
        dot_file = f"{out_name}.dot"
        with open(dot_file, "w") as f:
            f.write(dot.source)
        print(f"\nGraphviz executable not found. Saved DOT source to: {dot_file}")
        print("To visualize:")
        print("  1. Install Graphviz: https://graphviz.org/download/")
        print(
            "  2. Or paste the content at: https://dreampuf.github.io/GraphvizOnline/"
        )
        return dot_file


def parse_math(s):
    # convert_xor lets users type ^ for power; implicit multiplication allows "2x" -> 2*x
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    return parse_expr(s, transformations=transformations)


def count_nodes(expr):
    """Count total nodes in the expression tree."""
    if not hasattr(expr, "args") or len(expr.args) == 0:
        return 1  # Leaf node
    return 1 + sum(count_nodes(child) for child in expr.args)


def tree_depth(expr):
    """Calculate the depth of the expression tree as a binary tree.

    SymPy represents operations like Add and Mul as n-ary (with n children),
    but we need to calculate depth assuming binary tree decomposition.
    For n-ary operations, we use left-associative grouping.

    Example: x1*y1 + x2*y2 + x3*y3 becomes ((x1*y1) + (x2*y2)) + (x3*y3)
    """
    if not hasattr(expr, "args") or len(expr.args) == 0:
        return 1  # Leaf node has depth 1

    children_depths = [tree_depth(child) for child in expr.args]
    n = len(children_depths)

    if n == 1:
        return 1 + children_depths[0]
    elif n == 2:
        return 1 + max(children_depths)
    else:  # n > 2: decompose into binary tree (left-associative)
        # The leftmost child goes through (n-2) additional binary nodes
        # e.g., a+b+c+d becomes (((a+b)+c)+d)
        return 1 + max(n - 2 + children_depths[0], max(children_depths[1:]))


if __name__ == "__main__":
    s = "sin(x) + (x^2)/(1+y)"  # <- replace with your expression
    expr = parse_math(s)
    print("Parsed as:", expr)

    # Calculate and print tree statistics
    num_nodes = count_nodes(expr)
    depth = tree_depth(expr)
    print(f"Number of nodes: {num_nodes}")
    print(f"Tree depth: {depth}")

    out_file = sympy_to_graphviz(expr, out_name="expr_tree", fmt="png")
    print("Wrote:", out_file)

    print("Visualize online at https://dreampuf.github.io/GraphvizOnline/")
