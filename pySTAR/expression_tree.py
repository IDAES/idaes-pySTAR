import logging
import pandas as pd
from pyomo.core.expr.sympy_tools import PyomoSympyBimap, sympy2pyomo_expression
import pyomo.environ as pyo
import sympy as sp

# pylint: disable = logging-fstring-interpolation
_logger = logging.getLogger(__name__)


UNARY_FUNC_MAP = {
    "exp": sp.exp,
    "log": sp.log,
    "square": lambda x: sp.Pow(x, 2),
    "sqrt": sp.sqrt,
}
BINARY_FUNC_MAP = {
    "sum": sp.Add,
    "mult": sp.Mul,
    "diff": lambda x, y: sp.Add(x, -y),
    "div": lambda x, y: sp.Mul(x, 1 / y),
}


class Node:
    """Template for a node in the expression tree"""

    def __init__(self, value, left_child=None, right_child=None, tree_index=None):
        self.value = value
        self.right_child = right_child
        self.left_child = left_child
        self._is_operand = False
        self._is_constant = False
        self._tree_index = tree_index
        self._construct_sympy_expression()
        self._enhanced_expr = None

    def __str__(self):
        return str(self.sympy_expression)

    @property
    def sympy_expression(self):
        """Returns sympy expression for the node"""
        return self._sympy_expr

    def is_operand(self):
        """Returns True if the node is an operand"""
        return self._is_operand

    def is_constant(self):
        """Returns True if the node is a constant"""
        return self._is_constant

    def _get_unary_expression(self, right_child_expr):
        try:
            return UNARY_FUNC_MAP[self.value](right_child_expr)
        except KeyError as excp:
            raise ValueError(f"Unrecognized unary operator {self.value}") from excp

    def _get_binary_expression(self, left_child_expr, right_child_expr):
        try:
            return BINARY_FUNC_MAP[self.value](left_child_expr, right_child_expr)
        except KeyError as excp:
            raise ValueError(f"Unrecognized binary operator {self.value}") from excp

    def _construct_sympy_expression(self):
        """Constructs sympy expression for the node"""
        # This is a leaf node/operand
        if self.right_child is None and self.left_child is None:
            if isinstance(self.value, str):
                self._sympy_expr = sp.symbols(self.value)
                if "c_" in self.value:
                    self._is_constant = True
                else:
                    self._is_operand = True
            else:
                # This must either be an integer or a float
                self._sympy_expr = self.value
                self._is_constant = True
            return

        # This is a unary operator
        if self.left_child is None:
            self._sympy_expr = self._get_unary_expression(
                self.right_child.sympy_expression
            )
            return

        # This is a binary operator
        self._sympy_expr = self._get_binary_expression(
            self.left_child.sympy_expression, self.right_child.sympy_expression
        )
        return

    def get_enhanced_expression(self, enhance_subtree: bool = True):
        """Returns the node expression with a slope and intercept"""
        if self._enhanced_expr is not None:
            # The expression is already constructed, so return
            return self._enhanced_expr

        slope = sp.symbols(f"m_{self._tree_index}")
        intercept = sp.symbols(f" k_{self._tree_index}")

        # This is a leaf node/operand
        if self.right_child is None and self.left_child is None:
            if self._is_constant:
                self._enhanced_expr = self._sympy_expr
            else:
                # This must be an operand
                self._enhanced_expr = intercept + slope * self._sympy_expr

        # This is a unary operator
        elif self.left_child is None:
            expr = self._get_unary_expression(
                self.right_child.get_enhanced_expression(enhance_subtree)
            )
            self._enhanced_expr = intercept + slope * expr if enhance_subtree else expr

        # This is a binary operator
        else:
            expr = self._get_binary_expression(
                self.left_child.get_enhanced_expression(enhance_subtree),
                self.right_child.get_enhanced_expression(enhance_subtree),
            )
            self._enhanced_expr = intercept + slope * expr if enhance_subtree else expr

        return self._enhanced_expr

    def unformatted_expression(self):
        """Returns the expression as is"""
        # NOTE: This method is probably not helpful. Need to delete this method
        # after finalizing the code.
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

    def __init__(self, tree: list | dict):
        if isinstance(tree, list):
            self.tree = dict(tree)
        else:  # This must be a dict
            self.tree = tree

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
                    tree_index=i,
                )

    def __str__(self):
        return str(self.nodes[1])

    def __add__(self, et):
        return self.sympy_expression + et.sympy_expression

    def __sub__(self, et):
        return self.sympy_expression - et.sympy_expression

    @property
    def sympy_expression(self):
        """Returns the sympy expression"""
        return self.nodes[1].sympy_expression

    @property
    def simplified_expression(self):
        """Returns the simplified expression"""
        return self.nodes[1].simplified_expression

    def get_enhanced_expression(self, enhance_subtree: bool = True):
        """Returns the enhanced expression"""
        # Once an enhanced subtree is constructed, it cannot be changed.
        # So, we reset the enhanced expression before calling the method
        for node in self.nodes.values():
            # pylint: disable = protected-access
            node._enhanced_expr = None
        return self.nodes[1].get_enhanced_expression(enhance_subtree)

    def get_pyomo_model(
        self,
        cst_vars: dict | None = None,
        enhance_tree: bool = False,
        enhance_subtree: bool = True,
    ):
        """Returns a model of the expression tree"""
        if enhance_tree:
            expr = self.get_enhanced_expression(enhance_subtree)
        else:
            expr = self.sympy_expression  # sympy expression

        sympy_vars = list(expr.free_symbols)
        psm = PyomoSympyBimap()  # Pyomo -> sympy variable map

        # Declare pyomo Vars for operands and constants
        m = pyo.ConcreteModel()
        for v in sympy_vars:
            if cst_vars is None or str(v) not in cst_vars:
                setattr(m, str(v), pyo.Var())
                pyomo_var = getattr(m, str(v))

            else:
                pyomo_var = cst_vars[str(v)]

            psm.sympy2pyomo[v] = pyomo_var
            psm.pyomo2sympy[pyomo_var] = v

        m.output_var = pyo.Var()
        m.calculate_output_var = pyo.Constraint(
            expr=m.output_var == sympy2pyomo_expression(expr=expr, object_map=psm)
        )

        return m

    def get_parameter_estimation_model(
        self,
        data: pd.DataFrame,
        enhance_tree: bool = False,
        enhance_subtree: bool = True,
    ):
        """Constructs the parameter estimation model for the given data"""
        if enhance_tree:
            expr = self.get_enhanced_expression(enhance_subtree)
        else:
            expr = self.sympy_expression  # sympy expression

        var_list = [str(v) for v in expr.free_symbols]
        param_list = [v for v in var_list if v not in data.columns]
        valid_var_list = [v for v in data.columns if v in var_list]
        invalid_var_list = [v for v in data.columns if v != "y" and v not in var_list]

        _logger.info(f"List of parameters in the expression: {param_list}")
        if len(invalid_var_list) > 0:
            _logger.warning(
                f"Variable(s) {invalid_var_list} is/are not found in the expression!"
            )

        m = pyo.ConcreteModel()
        # Define variables for parameters in the main model
        for v in param_list:
            setattr(m, v, pyo.Var())

        # Construct a pyomo model of the expression for cloning
        cst_vars = {v: getattr(m, v) for v in param_list}
        expr_model = self.get_pyomo_model(cst_vars, enhance_tree, enhance_subtree)

        # Construct the expression for each sample
        m.sample = pyo.Block(data.index.to_list())
        for s in m.sample:
            # Declare variables and expression using the clone method
            m.sample[s].transfer_attributes_from(expr_model.clone())
            blk = m.sample[s]  # Pointer to the sample model

            # Fix variable values
            for v in valid_var_list:
                getattr(blk, v).fix(data.loc[s, v])

            # Calculate residual
            blk.residual = pyo.Var()
            blk.calculate_residual = pyo.Constraint(
                expr=blk.residual == blk.output_var - data.loc[s, "y"]
            )
            blk.square_of_residual = pyo.Expression(expr=blk.residual**2)

        # Add an objective
        m.sse = pyo.Objective(expr=sum(m.sample[:].square_of_residual))

        return m
