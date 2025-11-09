from itertools import product
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import Var, Constraint

# pylint: disable = import-error
from bigm_operators import BigmSampleBlock, BigmSampleBlockData
from hull_operators import HullSampleBlock

LOGGER = logging.getLogger(__name__)
SUPPORTED_UNARY_OPS = ["square", "sqrt", "log", "exp"]
SUPPORTED_BINARY_OPS = ["sum", "diff", "mult", "div"]
SUPPORTED_UNARY_OPS += ["exp_old", "exp_comp", "log_old", "log_comp"]


# pylint: disable = logging-fstring-interpolation
class SymbolicRegressionModel(pyo.ConcreteModel):
    """Builds the symbolic regression model for a given data"""

    def __init__(
        self,
        *args,
        data: pd.DataFrame,
        input_columns: List[str],
        output_column: str,
        tree_depth: int,
        operators: Optional[list] = None,
        var_bounds: tuple = (-100, 100),
        constant_bounds: tuple = (-100, 100),
        eps: float = 1e-4,
        model_type: str = "bigm",
        **kwds,
    ):
        super().__init__(*args, **kwds)

        # Perform input data checks
        if tree_depth < 2:
            raise ValueError(f"tree_depth must be >= 2. Received {tree_depth}.")

        # Check if the received operators are supported or not
        if operators is None:
            # If not specified, use all supported operators
            unary_operators = SUPPORTED_UNARY_OPS
            binary_operators = SUPPORTED_BINARY_OPS

        else:
            # Ensure that there are no duplicates in the list of operators
            if len(operators) != len(set(operators)):
                raise ValueError("Duplicates are present in the list of operators")

            unary_operators = []
            binary_operators = []
            for op in operators:
                if op in SUPPORTED_UNARY_OPS:
                    unary_operators.append(op)
                elif op in SUPPORTED_BINARY_OPS:
                    binary_operators.append(op)
                else:
                    raise ValueError(f"Operator {op} is not recognized!")

        LOGGER.info(f"Using {unary_operators} unary operators.")
        LOGGER.info(f"Using {binary_operators} binary operators.")

        # Save a reference to the input and output data
        _col_name_map = {
            _name: f"x{index + 1}" for index, _name in enumerate(input_columns)
        }
        self.input_data_ref = data[input_columns].rename(columns=_col_name_map)
        self.output_data_ref = data[output_column]
        self.var_bounds = pyo.Param(
            ["lb", "ub"],
            initialize={"lb": var_bounds[0], "ub": var_bounds[1]},
        )
        self.constant_bounds = pyo.Param(
            ["lb", "ub"],
            initialize={"lb": constant_bounds[0], "ub": constant_bounds[1]},
        )
        self.eps_value = pyo.Param(initialize=eps, domain=pyo.PositiveReals)
        self.min_tree_size = pyo.Param(
            initialize=1,
            mutable=True,
            within=pyo.PositiveIntegers,
            doc="Minimum number of active nodes in the tree",
        )
        self.max_tree_size = pyo.Param(
            initialize=2**tree_depth,
            mutable=True,
            within=pyo.PositiveIntegers,
            doc="Maximum number of active nodes in the tree",
        )

        # Define model sets
        # pylint: disable = no-member
        self.binary_operators_set = pyo.Set(initialize=binary_operators)
        self.unary_operators_set = pyo.Set(initialize=unary_operators)
        self.operators_set = self.binary_operators_set.union(self.unary_operators_set)
        self.operands_set = pyo.Set(initialize=list(_col_name_map.values()) + ["cst"])
        self.collective_operator_set = self.operators_set.union(self.operands_set)

        # pre_non_terminal_nodes_set: Set of nodes without the last two layers
        self.pre_non_terminal_nodes_set = pyo.RangeSet(2 ** (tree_depth - 2) - 1)
        self.non_terminal_nodes_set = pyo.RangeSet(2 ** (tree_depth - 1) - 1)
        self.terminal_nodes_set = pyo.RangeSet(2 ** (tree_depth - 1), 2**tree_depth - 1)
        self.nodes_set = self.non_terminal_nodes_set.union(self.terminal_nodes_set)

        # Build the expression model
        self._build_expression_tree_model()

        # Calculate the output value for each sample
        if model_type == "bigm":
            self.samples = BigmSampleBlock(self.input_data_ref.index.to_list())
        elif model_type == "hull":
            self.samples = HullSampleBlock(self.input_data_ref.index.to_list())
        else:
            raise ValueError(f"model_type: {model_type} is not recognized.")

        # Build useful expressions
        self.sum_square_residual = pyo.Expression(
            expr=sum(self.samples[:].square_of_residual),
            doc="Evaluates the sum of square of errors across all samples",
        )

    @property
    def max_depth(self):
        """Returns the maximum possible depth of a tree"""
        return round(np.log2(len(self.nodes_set) + 1))

    @property
    def model_type(self):
        """Returns the type (bigm or hull) used to transform disjunctions"""
        first_sample_name = self.input_data_ref.iloc[0].name
        if isinstance(self.samples[first_sample_name], BigmSampleBlockData):
            return "bigm"

        return "hull"

    def _build_expression_tree_model(self):
        """Builds constraints that return a valid expression tree"""
        # Define variables
        # pylint: disable = attribute-defined-outside-init, not-an-iterable
        self.select_node = Var(
            self.nodes_set,
            domain=pyo.Binary,
            doc="If 1, an operator/operator is assigned to the node",
        )
        self.select_operator = Var(
            self.nodes_set,
            self.collective_operator_set,
            domain=pyo.Binary,
            doc="Binary variable associated with the operator at each node",
        )
        self.constant_val = Var(self.nodes_set, doc="Constant values in the expression")

        # Binary and unary operators must not be chosen for terminal nodes
        for n in self.terminal_nodes_set:
            for op in self.operators_set:
                self.select_operator[n, op].fix(0)

        # Constant must only be present in the left node (i.e., even-numbered nodes)
        for n in self.nodes_set:
            if n % 2 == 1:
                self.select_operator[n, "cst"].fix(0)

        # Begin constructing essential constraints
        # If a node is active, then either an operator or an operand
        # must be chosen
        @self.Constraint(self.nodes_set)
        def active_node_active_operator(blk, n):
            return blk.select_node[n] == sum(
                blk.select_operator[n, op] for op in self.collective_operator_set
            )

        # If a binary or unary operator is chosen, then right child must be present
        @self.Constraint(self.non_terminal_nodes_set)
        def right_child_presence(blk, n):
            return blk.select_node[2 * n + 1] == sum(
                blk.select_operator[n, op] for op in self.operators_set
            )

        # If a binary operator is chosen, then left child must be present
        @self.Constraint(self.non_terminal_nodes_set)
        def left_child_presence(blk, n):
            return blk.select_node[2 * n] == sum(
                blk.select_operator[n, op] for op in self.binary_operators_set
            )

        # Set constant value to zero, if constant operator is not selected
        @self.Constraint(self.nodes_set)
        def constant_lb_con(blk, n):
            return (
                blk.constant_bounds["lb"] * blk.select_operator[n, "cst"]
                <= blk.constant_val[n]
            )

        @self.Constraint(self.nodes_set)
        def constant_ub_con(blk, n):
            return (
                blk.constant_val[n]
                <= blk.constant_bounds["ub"] * blk.select_operator[n, "cst"]
            )

    # pylint: disable = attribute-defined-outside-init
    def add_objective(self, objective_type: str = "sse"):
        """Appends objective function to the model

        Parameters
        ----------
        objective_type : str, optional
            Choice of objective function. Supported objective functions are
            Sum of squares of errors: "sse"
            Bayesian Information Criterion: "bic", by default "sse"
        """
        if objective_type == "sse":
            # build SSR objective
            self.sse = pyo.Objective(expr=self.sum_square_residual)

        elif objective_type == "bic":
            # Build BIC objective
            raise NotImplementedError("BIC is not supported currently")

        else:
            raise ValueError(
                f"Specified objective_type: {objective_type} is not supported."
            )

    def add_same_operand_operation_cuts(self):
        """Adds cuts to avoid expressions of type: x + x, x - x, x * x, x / x"""

        @self.Constraint(self.non_terminal_nodes_set, self.operands_set)
        def redundant_bin_op_same_operand(blk, n, op):
            if op == "cst":
                # Both child nodes cannot take a constant, so skip this case
                return Constraint.Skip

            # Do not choose the same operand for both the child nodes.
            return (
                blk.select_operator[2 * n, op] + blk.select_operator[2 * n + 1, op]
                <= blk.select_node[n]
            )

    def add_constant_operation_cuts(self):
        """
        Adds constraints to remove expressions of type: cst (op_1) (cst (op_2) x1)
        """
        op_list = []
        if "sum" in self.binary_operators_set and "diff" in self.binary_operators_set:
            op_list += list(product(["sum", "diff"], ["sum", "diff"]))

        if "mult" in self.binary_operators_set and "div" in self.binary_operators_set:
            op_list += list(product(["mult", "div"], ["mult", "div"]))

        @self.Constraint(self.pre_non_terminal_nodes_set, op_list)
        def redundant_cst_operations(blk, n, op1, op2):
            # cst[2 * n] op1[n] (cst[4 * n + 2] op1[2 * n + 1] ....)
            return (
                blk.select_operator[2 * n, "cst"]
                + blk.select_operator[4 * n + 2, "cst"]
                <= 3 * blk.select_node[n]
                - blk.select_operator[n, op1]
                - blk.select_operator[2 * n + 1, op2]
            )

    def add_associative_operation_cuts(self):
        """
        Adds cuts to remove associative operator combinations.
        A + B - C and A - (C - B) are equivalent, so remove former.
        A * (B / C) and A / (C / B) are equivalent, so remove former
        """
        op_list = []
        if "sum" in self.binary_operators_set and "diff" in self.binary_operators_set:
            op_list += [("sum", "diff")]

        if "mult" in self.binary_operators_set and "div" in self.binary_operators_set:
            op_list += [("mult", "div")]

        @self.Constraint(self.pre_non_terminal_nodes_set, op_list)
        def redundant_associative_operations(blk, n, op1, op2):
            # A + B - C and A - (C - B) are equivalent, so remove former
            # A * (B / C) and A / (C / B) are equivalent, so remove former
            return (
                blk.select_operator[n, op1] + blk.select_operator[2 * n + 1, op2]
                <= blk.select_node[n]
            )

    def add_inverse_function_composition_cuts(self):
        """
        Adds cuts to remove composition of inverse functions: exp(log(.)); square(sqrt(.))
        """

        def _inverse_function_rule(blk, n, op_1, op_2):
            return (
                blk.select_operator[n, op_1] + blk.select_operator[2 * n + 1, op_2]
                <= blk.select_node[n]
            )

        if "exp" in self.unary_operators_set and "log" in self.unary_operators_set:
            self.redundant_inv_op_exp_log = Constraint(
                self.pre_non_terminal_nodes_set,
                [("exp", "log"), ("log", "exp")],
                rule=_inverse_function_rule,
            )

        if "square" in self.unary_operators_set and "sqrt" in self.unary_operators_set:
            self.redundant_inv_op_square_sqrt = Constraint(
                self.pre_non_terminal_nodes_set,
                [("square", "sqrt"), ("sqrt", "square")],
                rule=_inverse_function_rule,
            )

    def add_symmetry_breaking_cuts(self, sample_name=None):
        """Adds cuts to eliminate symmetrical trees from the model"""
        if sample_name is None:
            # If the sample is not specified, apply symmetry breaking
            # cuts to the first sample
            sample_name = self.input_data_ref.iloc[0].name

        self.samples[sample_name].add_symmetry_breaking_cuts()

    def add_implication_cuts(self):
        """
        Adds cuts to avoid a certain combinations of operators based on the input data
        """
        # Avoid division by zero
        near_zero_operands, negative_operands = _get_operand_domains(
            data=self.input_data_ref, tol=pyo.value(self.eps_value)
        )

        # pylint: disable = logging-fstring-interpolation
        if len(near_zero_operands) > 0 and "div" in self.binary_operators_set:
            LOGGER.info(
                f"Data for operands {near_zero_operands} is close to zero. "
                f"Adding cuts to prevent division with these operands."
            )

            @self.Constraint(self.non_terminal_nodes_set, near_zero_operands)
            def implication_cuts_div_operator(blk, n, op):
                return (
                    blk.select_operator[n, "div"] + blk.select_operator[2 * n + 1, op]
                    <= blk.select_node[n]
                )

        if len(negative_operands) > 0 and "sqrt" in self.unary_operators_set:
            LOGGER.info(
                f"Data for operands {negative_operands} is negative. "
                f"Adding cuts to prevent sqrt of these operands."
            )

            @self.Constraint(self.non_terminal_nodes_set, negative_operands)
            def implication_cuts_sqrt_operator(blk, n, op):
                return (
                    blk.select_operator[n, "sqrt"] + blk.select_operator[2 * n + 1, op]
                    <= blk.select_node[n]
                )

        non_positive_operands = list(set(near_zero_operands + negative_operands))
        if len(non_positive_operands) > 0 and "log" in self.unary_operators_set:
            LOGGER.info(
                f"Data for operands {non_positive_operands} is non-positive. "
                f"Adding cuts to prevent log of these operands."
            )

            @self.Constraint(self.non_terminal_nodes_set, non_positive_operands)
            def implication_cuts_log_operator(blk, n, op):
                return (
                    blk.select_operator[n, "log"] + blk.select_operator[2 * n + 1, op]
                    <= blk.select_node[n]
                )

    def constrain_max_tree_size(self, size: int):
        """Adds a constraint to constrain the maximum size of the tree"""
        self.max_tree_size = size

        # If the constraint exists, then updating the parameter value is sufficient
        # pylint: disable = attribute-defined-outside-init
        if not hasattr(self, "max_tree_size_constraint"):
            self.max_tree_size_constraint = Constraint(
                expr=sum(self.select_node[n] for n in self.nodes_set)
                <= self.max_tree_size
            )

    def constrain_min_tree_size(self, size: int):
        """Adds a constraint to constrain the minimum size of the tree"""
        self.min_tree_size = size

        # If the constraint exists, then updating the parameter value is sufficient
        # pylint: disable = attribute-defined-outside-init
        if not hasattr(self, "min_tree_size_constraint"):
            self.min_tree_size_constraint = Constraint(
                expr=sum(self.select_node[n] for n in self.nodes_set)
                >= self.min_tree_size
            )

    def relax_integrality_constraints(self):
        """Relaxes integrality requirement on all binary variables"""
        self.select_node.domain = pyo.UnitInterval
        self.select_operator.domain = pyo.UnitInterval

    def add_integrality_constraints(self):
        """Adds integrality requirement on node and operator variables"""
        self.select_node.domain = pyo.Binary
        self.select_operator.domain = pyo.Binary

    def relax_nonconvex_constraints(self):
        """Convexifies all non-convex nonlinear constraints"""
        raise NotImplementedError("Model Convexification is not currently supported")

    def get_selected_operators(self):
        """
        Returns the list of selected nodes along with the selected
        operator at that node
        """
        return [op for op, val in self.select_operator.items() if val.value > 0.95]

    def selected_tree_to_expression(self):
        """Returns the optimal expression as a string"""

    def get_parity_plot_data(self):
        """Returns a DataFrame containing actual outputs and predicted outputs"""
        results = pd.DataFrame()
        results["sim_data"] = self.output_data_ref

        if self.model_type == "bigm":
            results["prediction"] = [
                self.samples[s].val_node[1].value for s in self.samples
            ]
        else:
            results["prediction"] = [
                self.samples[s].node[1].val_node.value for s in self.samples
            ]

        results["square_of_error"] = [
            self.samples[s].square_of_residual.expr() for s in self.samples
        ]

        return results


def _get_operand_domains(data: pd.DataFrame, tol: float):
    near_zero_operands = []
    negative_operands = []

    for op in data.columns:
        if (abs(data[op]) < tol).any():
            near_zero_operands.append(op)

        if (data[op] < 0).any():
            negative_operands.append(op)

    return near_zero_operands, negative_operands
