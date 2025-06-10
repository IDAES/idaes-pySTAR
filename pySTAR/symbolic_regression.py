from typing import Optional
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import Var, Constraint
from pySTAR.operators.operators import BaseOperatorData, SampleBlock


class SymbolicRegressionModel(pyo.ConcreteModel):
    """Builds the symbolic regression model for a given data"""

    def __init__(
        self,
        *args,
        data: pd.DataFrame,
        input_columns: list,
        output_column: str,
        tree_depth: int,
        unary_operators: Optional[list] = None,
        binary_operators: Optional[list] = None,
        var_bounds: tuple = (-100, 100),
        constant_bounds: tuple = (-100, 100),
        eps: float = 1e-4,
        **kwds,
    ):
        super().__init__(*args, **kwds)

        if unary_operators is None:
            # If not specified, use all supported unary operators
            unary_operators = ["square", "sqrt", "exp", "log"]

        if binary_operators is None:
            # If not specified, use all supported binary operators
            binary_operators = ["sum", "diff", "mult", "div"]

        # Save a reference to the input and output data
        self.input_data_ref = data[input_columns]
        self.output_data_ref = data[output_column]
        self.var_bounds = pyo.Param(
            ["lb", "ub"],
            initialize={"lb": var_bounds[0], "ub": var_bounds[1]},
        )
        self.constant_bounds = pyo.Param(
            ["lb", "ub"],
            initialize={"lb": constant_bounds[0], "ub": constant_bounds[1]},
            mutable=True,
        )
        self.eps_val = pyo.Param(initialize=eps, mutable=True, domain=pyo.PositiveReals)
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
        self.operands_set = pyo.Set(initialize=input_columns + ["cst"])
        self.collective_operator_set = self.operators_set.union(self.operands_set)

        self.non_terminal_nodes_set = pyo.RangeSet(2 ** (tree_depth - 1) - 1)
        self.terminal_nodes_set = pyo.RangeSet(2 ** (tree_depth - 1), 2**tree_depth - 1)
        self.nodes_set = self.non_terminal_nodes_set.union(self.terminal_nodes_set)

        # Build the expression model
        self._build_expression_tree_model()

        # Calculate the output value for each sample
        self.samples = SampleBlock(self.input_data_ref.index.to_list())

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
    def add_objective(self, objective_type="sse"):
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
            self.sse = pyo.Objective(expr=sum(self.samples[:].square_of_error))

        elif objective_type == "bic":
            # Build BIC objective
            pass

        else:
            raise ValueError(
                f"Specified objective_type: {objective_type} is not supported."
            )

    def add_redundancy_cuts(self):
        """Adds cuts to eliminate redundance from the model"""

        # Do not choose the same operand for both the child nodes.
        # This cut avoids expressions of type: x + x, x - x, x * x, x / x.
        @self.Constraint(self.non_terminal_nodes_set, self.operands_set)
        def redundant_bin_op_same_operand(blk, n, op):
            if op == "cst":
                # Both child nodes cannot take a constant, so skip this case
                return Constraint.Skip

            return (
                blk.select_operator[2 * n, op] + blk.select_operator[2 * n + 1, op]
                <= blk.select_node[n]
            )

        # Remove Associative operator combinations

    def add_symmetry_breaking_cuts(self, sample_name=None):
        """Adds cuts to eliminate symmetrical trees from the model"""
        if sample_name is None:
            # If the sample is not specified, apply symmetry breaking
            # cuts to the first sample
            sample_name = self.input_data_ref.iloc[0].name

        blk = self.samples[sample_name]
        symmetric_operators = [
            op for op in ["sum", "mult"] if op in self.binary_operators_set
        ]

    def add_implication_cuts(self):
        """
        Adds cuts to avoid a certain combinations of operators based on the input data
        """

    def add_tree_size_constraint(self, size: int):
        """Adds a constraint to constrain the size of the tree"""
        self.max_tree_size = size

        # If the constraint exists, then updating the parameter value is sufficient
        # pylint: disable = attribute-defined-outside-init
        if not hasattr(self, "tree_size_con"):
            self.tree_size_constraint = Constraint(
                expr=sum(self.select_node[n] for n in self.nodes_set)
                <= self.max_tree_size
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
        for blk in self.component_data_objects(pyo.Block):
            if isinstance(blk, BaseOperatorData):
                blk.construct_convex_relaxation()

    def get_selected_operators(self):
        """
        Returns the list of selected nodes along with the selected
        operator at that node
        """
        return [op for op, val in self.select_operator.items() if val.value > 0.95]

    def selected_tree_to_expression(self):
        """Returns the optimal expression as a string"""
