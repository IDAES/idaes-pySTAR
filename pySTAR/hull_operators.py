"""
This module contains all the supported operator models
for symbolic regression.
"""

import logging
import pandas as pd
from pyomo.core.base.block import BlockData, declare_custom_block
from pyomo.environ import Var, Constraint
import pyomo.environ as pyo

LOGGER = logging.getLogger(__name__)


@declare_custom_block("BaseOperator", rule="build")
class BaseOperatorData(BlockData):
    """Base class for defining operator models"""

    _is_unary_operator = False
    _define_aux_var_right = False

    @property
    def symbolic_regression_model(self):
        """Returns a pointer to the symbolic regression model"""
        return self._bin_var_ref.parent_block()

    @property
    def operator_binary(self):
        """Returns the binary variable associated with the operator"""
        return self._bin_var_ref

    # pylint: disable = attribute-defined-outside-init, unused-argument
    def build(self, *args, bin_var: Var):
        """rule for constructing operator blocks"""
        srm = bin_var.parent_block()  # Symbolic regression model
        lb = srm.var_bounds["lb"]
        ub = srm.var_bounds["ub"]
        eps = srm.eps_value

        # Declare auxiliary variables
        self._bin_var_ref = bin_var  # Store a reference to the binary var
        self.val_node = Var(doc="Node value", bounds=(lb, ub))
        self.val_left_node = Var(doc="Left child value", bounds=(lb, ub))
        self.val_right_node = Var(doc="Right child value", bounds=(lb, ub))

        if self._is_unary_operator:
            # For unary operators, left node must not be chosen
            self.val_left_node.fix(0)

        if self._define_aux_var_right:
            # Declare an auxiliary variable for operators with singularity issue
            self.aux_var_right = Var(
                doc="Auxiliary variable for the right child",
                bounds=(eps, max(1, ub)),  # Ensure that the RHS is strictly positive
            )

            # Ensure that aux_var_right = (val_right_node if bin_var = 1 else 1)
            self.calculate_aux_var_right = Constraint(
                expr=self.aux_var_right == self.val_right_node + 1 - bin_var
            )

        # Add operator-specific constraints
        try:
            self.build_operator_model()
        except NotImplementedError:
            LOGGER.warning("Operator-specific model constraints are not implemented.")

    def build_operator_model(self):
        """Adds operator-specific constraints"""
        raise NotImplementedError("Operator-specific model is not implemented")

    # pylint: disable = no-member, attribute-defined-outside-init
    def add_bound_constraints(self, val_left_node=True, val_right_node=True):
        """Adds bound constraints on val_node variable"""
        bin_var = self._bin_var_ref

        self.node_val_lb_con = Constraint(
            expr=self.val_node.lb * bin_var <= self.val_node
        )
        self.node_val_ub_con = Constraint(
            expr=self.val_node <= self.val_node.ub * bin_var
        )

        if val_left_node:
            self.left_node_val_lb_con = Constraint(
                expr=self.val_left_node.lb * bin_var <= self.val_left_node
            )
            self.left_node_val_ub_con = Constraint(
                expr=self.val_left_node <= self.val_left_node.ub * bin_var
            )

        if val_right_node:
            self.right_node_val_lb_con = Constraint(
                expr=self.val_right_node.lb * bin_var <= self.val_right_node
            )
            self.right_node_val_ub_con = Constraint(
                expr=self.val_right_node <= self.val_right_node.ub * bin_var
            )

    def construct_convex_relaxation(self):
        """Constructs a convex relaxation of the model"""
        raise NotImplementedError("Convex relaxation is not supported")

    @staticmethod
    def compute_node_value(left_child, right_child):
        """Helper function to evaluate the value for a given operator"""
        raise NotImplementedError("Helper function is not implemented")


# pylint: disable = attribute-defined-outside-init, missing-class-docstring
@declare_custom_block("SumOperator", rule="build")
class SumOperatorData(BaseOperatorData):

    def build_operator_model(self):
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self.val_left_node + self.val_right_node
        )
        self.add_bound_constraints()

    def construct_convex_relaxation(self):
        # Operator model is convex by default, so return.
        pass

    @staticmethod
    def compute_node_value(left_child, right_child):
        return left_child + right_child


@declare_custom_block("DiffOperator", rule="build")
class DiffOperatorData(BaseOperatorData):

    def build_operator_model(self):
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self.val_left_node - self.val_right_node
        )
        self.add_bound_constraints()

    def construct_convex_relaxation(self):
        # Operator model is convex by default, so return.
        pass

    @staticmethod
    def compute_node_value(left_child, right_child):
        return left_child - right_child


@declare_custom_block("MultOperator", rule="build")
class MultOperatorData(BaseOperatorData):

    def build_operator_model(self):
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self.val_left_node * self.val_right_node
        )
        self.add_bound_constraints()

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(left_child, right_child):
        return left_child * right_child


@declare_custom_block("DivOperator", rule="build")
class DivOperatorData(BaseOperatorData):
    _define_aux_var_right = True

    def build_operator_model(self):
        self.evaluate_val_node = Constraint(
            expr=self.val_node * self.aux_var_right == self.val_left_node
        )
        self.add_bound_constraints()

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(left_child, right_child):
        return left_child / right_child


# pylint: disable = no-member
@declare_custom_block("SquareOperator", rule="build")
class SquareOperatorData(BaseOperatorData):
    _is_unary_operator = True

    def build_operator_model(self):
        # Evaluate the value at the node
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self.val_right_node * self.val_right_node
        )

        # val_node will be non-negative in this case, so update lb
        self.val_node.setlb(0)

        # For unary operator, left node is fixed to zero.
        self.add_bound_constraints(val_left_node=False)

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return right_child**2


@declare_custom_block("SqrtOperator", rule="build")
class SqrtOperatorData(BaseOperatorData):
    _is_unary_operator = True

    def build_operator_model(self):
        # Expressing the constraint in this manner makes it a
        # quadratic constraint, instead of a general nonlinear constraint
        self.evaluate_val_node = Constraint(
            expr=self.val_node * self.val_node == self.val_right_node
        )

        # val_right_node must be non-negative in this case
        self.val_right_node.setlb(0)
        self.add_bound_constraints(val_left_node=False)

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.sqrt(right_child)


@declare_custom_block("ExpOperator", rule="build")
class ExpOperatorData(BaseOperatorData):
    _is_unary_operator = True

    def build_operator_model(self):
        # Update the domain of variables
        ub_val_node = self.val_node.ub
        ub_val_right_node = self.val_right_node.ub

        self.val_right_node.setub(min(pyo.log(ub_val_node), ub_val_right_node))
        self.val_node.setlb(0)

        # To avoid numerical issues, we do not let the lower bound of the
        # argument of the exp function go below -10
        self.val_right_node.setlb(max(-10, self.val_right_node.lb))
        self.add_bound_constraints(val_left_node=False)

        # If the operator is not chosen, then val_right_node = 0,
        # exp(val_right_node) = 1. So, we add (bin_var - 1) to make the
        # expression evaluate to zero.
        self.evaluate_val_node = Constraint(
            expr=self.val_node == pyo.exp(self.val_right_node) + self._bin_var_ref - 1
        )

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.exp(right_child)


@declare_custom_block("LogOperator", rule="build")
class LogOperatorData(BaseOperatorData):
    _is_unary_operator = True
    _define_aux_var_right = True

    def build_operator_model(self):
        # If the operator is not selected, aux_var_right = 1, so log vanishes
        self.evaluate_val_node = Constraint(
            expr=self.val_node == pyo.log(self.aux_var_right)
        )

        self.add_bound_constraints(val_left_node=False)

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.log(right_child)


OPERATOR_MODELS = {
    "sum": SumOperator,
    "diff": DiffOperator,
    "mult": MultOperator,
    "div": DivOperator,
    "square": SquareOperator,
    "sqrt": SqrtOperator,
    "log": LogOperator,
    "exp": ExpOperator,
}


@declare_custom_block("HullSampleBlock", rule="build")
class HullSampleBlockData(BlockData):
    """Class for evaluating the expression tree for each sample"""

    @property
    def symbolic_regression_model(self):
        """Returns a pointer to the symbolic regression model"""
        return self.parent_block()

    def build(self, s):
        """rule for building the expression model for each sample"""
        pb = self.parent_block()
        input_data = pb.input_data_ref.loc[s]
        output_value = pb.output_data_ref.loc[s]

        @self.Block(pb.nodes_set)
        def node(blk, n):
            # Declare a variable to track the value at this node
            blk.val_node = Var(
                doc="Value at the node",
                bounds=(pb.var_bounds["lb"], pb.var_bounds["ub"]),
            )

            # Defining an expression for the val_node variable arising from operands
            blk.value_from_operands = pyo.Expression(
                expr=sum(
                    pb.select_operator[n, op] * input_data[op]
                    for op in input_data.index
                )
                + pb.constant_val[n]
            )

            # Define a constraint to evaluate the value from all disjuncts
            if n in pb.terminal_nodes_set:
                # For terminal nodes, val_node = value from the operands chosen
                blk.evaluate_val_node_var = Constraint(
                    expr=blk.val_node == blk.value_from_operands
                )

        for n in pb.non_terminal_nodes_set:
            op_blocks = []
            for op in pb.operators_set:
                # Build operator blocks for all unary and binary operators
                setattr(
                    self.node[n],
                    op + "_operator",
                    OPERATOR_MODELS[op](bin_var=pb.select_operator[n, op]),
                )
                op_blocks.append(getattr(self.node[n], op + "_operator"))

            self.node[n].evaluate_val_node_var = Constraint(
                expr=self.node[n].val_node
                == sum(blk.val_node for blk in op_blocks)
                + self.node[n].value_from_operands
            )

            self.node[n].evaluate_val_left_node = Constraint(
                expr=self.node[2 * n].val_node
                == sum(blk.val_left_node for blk in op_blocks)
            )
            self.node[n].evaluate_val_right_node = Constraint(
                expr=self.node[2 * n + 1].val_node
                == sum(blk.val_right_node for blk in op_blocks)
            )

        self.residual = pyo.Var(doc="Residual value for the sample")
        self.calculate_residual = pyo.Constraint(
            expr=self.residual == output_value - self.node[1].val_node,
            doc="Computes the residual between prediction and the data",
        )
        self.square_of_residual = pyo.Expression(expr=self.residual**2)

    def add_symmetry_breaking_cuts(self):
        """Adds symmetry breaking cuts to the sample"""
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]
        symmetric_operators = [
            op for op in ["sum", "mult"] if op in srm.binary_operators_set
        ]

        @self.Constraint(srm.non_terminal_nodes_set)
        def symmetry_breaking_constraints(blk, n):
            return blk.node[2 * n].val_node - blk.node[2 * n + 1].val_node >= (
                vlb - vub
            ) * (
                srm.select_node[n]
                - sum(srm.select_operator[n, op] for op in symmetric_operators)
            )

    def compare_node_values(self):
        """Compares the node values obtained from the model and manual calculation"""
        srm = self.symbolic_regression_model
        data = srm.input_data_ref.loc[self.index()]
        true_value = {
            n: srm.constant_val[n].value
            + sum(
                data[op] * srm.select_operator[n, op].value
                for op in srm.operands_set
                if op != "cst"
            )
            for n in srm.nodes_set
        }
        for n in range(len(srm.non_terminal_nodes_set), 0, -1):
            for op in srm.operators_set:
                if srm.select_operator[n, op].value > 0.99:
                    # Operator is selector
                    true_value[n] += getattr(
                        self.node[n], op + "_operator"
                    ).compute_node_value(true_value[2 * n], true_value[2 * n + 1])

        computed_value = {n: self.node[n].val_node.value for n in self.node}

        return pd.DataFrame.from_dict(
            {"True Value": true_value, "Computed Value": computed_value}
        )
