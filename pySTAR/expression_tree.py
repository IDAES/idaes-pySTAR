from pyomo.core.expr.sympy_tools import PyomoSympyBimap, sympy2pyomo_expression
import pyomo.environ as pyo
import sympy as sp


class Node:
    """Template for a node in the expression tree"""

    def __init__(self, value, left_child=None, right_child=None):
        self.value = value
        self.right_child = right_child
        self.left_child = left_child
        self._construct_sympy_expression()
        self._is_operand = False

    def __str__(self):
        return str(self.sympy_expression)

    def is_operand(self):
        """Returns True if the node is an operand"""
        return self._is_operand

    @property
    def sympy_expression(self):
        """Returns sympy expression for the node"""
        return self._sympy_expr

    def _construct_sympy_expression(self):
        """Constructs sympy expression for the node"""
        # This is a leaf node/operand
        if self.right_child is None and self.left_child is None:
            if isinstance(self.value, str):
                self._sympy_expr = sp.symbols(self.value)
                self._is_operand = True
            else:
                # This must either be an integer or a float
                self._sympy_expr = self.value
            return

        # This is a unary operator
        if self.left_child is None:
            if self.value not in ["exp", "log", "square", "sqrt"]:
                raise ValueError(f"Unrecognized unary operator {self.value}")

            right_child_expr = self.right_child.sympy_expression
            expr = {
                "exp": sp.exp(right_child_expr),
                "log": sp.log(right_child_expr),
                "square": right_child_expr**2,
                "sqrt": sp.sqrt(right_child_expr),
            }
            self._sympy_expr = expr[self.value]
            return

        # This is a binary operator
        right_child_expr = self.right_child.sympy_expression
        left_child_expr = self.left_child.sympy_expression
        if self.value not in ["sum", "diff", "mult", "div"]:
            raise ValueError(f"Unrecognized binary operator {self.value}")

        expr = {
            "sum": left_child_expr + right_child_expr,
            "diff": left_child_expr - right_child_expr,
            "mult": left_child_expr * right_child_expr,
            "div": left_child_expr / right_child_expr,
        }
        self._sympy_expr = expr[self.value]
        return

    @property
    def simplified_expression(self):
        """Returns simplified expression"""
        return sp.simplify(self.sympy_expression)

    def unformatted_expression(self):
        """Returns the expression as is"""
        # This is a leaf node/operand
        if self.right_child is None and self.left_child is None:
            return f"{self.value}"

        # This is a unary operator
        if self.left_child is None:
            if self.value not in ["exp", "log", "square", "sqrt"]:
                raise ValueError(f"Unrecognized unary operator {self.value}")

            return f"{self.value}({self.right_child})"

        # This is a binary operator
        binary_ops = {"sum": "+", "diff": "-", "mult": "*", "div": "/"}
        if self.value not in binary_ops:
            raise ValueError(f"Unrecognized binary operator {self.value}")

        return f"({self.left_child} {binary_ops[self.value]} {self.right_child})"


class ExpressionTree:
    """Constructs a binary tree for a given expression"""

    def __init__(self, tree: list):
        self.tree = dict(tree)
        # Add subscripts to constants
        for _node, _op in self.tree.items():
            if _op == "cst":
                self.tree[_node] = f"c_{_node}"

        self.nodes = {}
        max_node = max(self.tree)
        for i in range(max_node, 0, -1):
            if i in self.tree:
                self.nodes[i] = Node(
                    value=self.tree[i],
                    left_child=self.nodes.get(2 * i, None),
                    right_child=self.nodes.get(2 * i + 1, None),
                )

    def __str__(self):
        return str(self.nodes[1])

    @property
    def sympy_expression(self):
        """Returns the sympy expression"""
        return self.nodes[1].sympy_expression

    @property
    def simplified_expression(self):
        """Returns the simplified expression"""
        return self.nodes[1].simplified_expression

    def get_pyomo_model(self):
        """Returns a model of the expression tree"""
        expr = self.sympy_expression  # sympy expression
        sympy_vars = list(expr.free_symbols)
        psm = PyomoSympyBimap()  # Pyomo -> sympy variable map

        # Declare pyomo Vars for operands and constants
        m = pyo.ConcreteModel()
        for v in sympy_vars:
            setattr(m, str(v), pyo.Var())
            pyomo_var = getattr(m, str(v))
            psm.sympy2pyomo[v] = pyomo_var
            psm.pyomo2sympy[pyomo_var] = v

        m.output_var = pyo.Var()
        m.calculate_output_var = pyo.Constraint(
            expr=m.output_var == sympy2pyomo_expression(expr=expr, object_map=psm)
        )

        return m
