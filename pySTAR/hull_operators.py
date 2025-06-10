"""
This module contains all the supported operator models
for symbolic regression.
"""

import logging
from pyomo.core.base.block import BlockData
from pyomo.environ import Var, Constraint
import pyomo.environ as pyo

from pySTAR.operators.custom_block import declare_custom_block
from pySTAR.operators.relaxation import mccormick_relaxation

LOGGER = logging.getLogger(__name__)


@declare_custom_block("BaseOperator")
class BaseOperatorData(BlockData):
    """Base class for defining operator models"""

    _operator_name = "base_operator"
    _is_unary_operator = False
    _define_aux_var_right = False

    @property
    def regression_model(self):
        """
        Returns the symbolic regression model

        Model structure:
        main_model -> Sample block -> Node block -> Operator block
        So, calling the parent_block method thrice returns the
        SymbolicRegressionModel object.
        """
        return self._bin_var_ref.parent_block()

    @property
    def operator_binary(self):
        """Returns the binary variable associated with the operator"""
        return self._bin_var_ref

    # pylint: disable = attribute-defined-outside-init
    def build(self, bin_var: Var):
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
                bounds=(eps, ub),  # Ensure that the RHS is strictly positive
            )

            # Define val_right_node = aux_var_right * binary_var
        self.calculate_val_right_node = Constraint(
            expr=self.val_right_node == bin_var * self.aux_var_right
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


# pylint: disable = attribute-defined-outside-init, missing-class-docstring
@declare_custom_block("SumOperator")
class SumOperatorData(BaseOperatorData):
    _operator_name = "sum"

    def build_operator_model(self):
        """Constructs constraint that evaluates the value at the node"""
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self.val_left_node + self.val_right_node
        )
        self.add_bound_constraints()

    def construct_convex_relaxation(self):
        # Operator model is convex by default, so return.
        pass


@declare_custom_block("DiffOperator")
class DiffOperatorData(BaseOperatorData):
    _operator_name = "diff"

    def build_operator_model(self):
        """Constructs constraint that evaluates the value at the node"""
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self.val_left_node - self.val_right_node
        )
        self.add_bound_constraints()

    def construct_convex_relaxation(self):
        # Operator model is convex by default, so return.
        pass


@declare_custom_block("MultOperator")
class MultOperatorData(BaseOperatorData):
    _operator_name = "mult"

    def build_operator_model(self):
        """Constructs constraint that evaluates the value at the node"""
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self.val_left_node * self.val_right_node
        )
        self.add_bound_constraints()

    def construct_convex_relaxation(self):
        self.evaluate_val_node.deactivate()
        mccormick_relaxation(
            self, z=self.val_node, x=self.val_left_node, y=self.val_right_node
        )


@declare_custom_block("DivOperator")
class DivOperatorData(BaseOperatorData):
    _operator_name = "div"
    _define_aux_var_right = True

    def build_operator_model(self):
        """Constructs constraint that evaluates the value at the node"""
        self.evaluate_val_node = Constraint(
            expr=self.val_node * self.aux_var_right == self.val_left_node
        )
        self.add_bound_constraints(val_right_node=False)

    def construct_convex_relaxation(self):
        self.evaluate_val_node.deactivate()
        self.calculate_val_right_node.deactivate()
        mccormick_relaxation(
            self, z=self.val_left_node, x=self.val_node, y=self.aux_var_right
        )
        mccormick_relaxation(
            self,
            z=self.val_right_node,
            x=self.get_operator_binary_var(),
            y=self.aux_var_right,
        )


# pylint: disable = no-member
@declare_custom_block("SquareOperator")
class SquareOperatorData(BaseOperatorData):
    _operator_name = "square"
    _is_unary_operator = True

    def build_operator_model(self):
        """Constructs constraint that evaluates the value at the node"""
        # For unary operator, left node is fixed to zero.
        self.add_bound_constraints(val_left_node=False)

        # Evaluate the value at the node
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self.val_right_node * self.val_right_node
        )

    def construct_convex_relaxation(self):
        # Need to implement outer-approximation
        # Update bounds on val_node and val_right_node variables
        ub_val_node = self.val_node.ub
        ub_val_right_node = self.val_right_node.ub
        lb_val_right_node = self.val_right_node.lb

        self.val_node.setub(min(ub_val_node, ub_val_right_node**2))
        self.val_right_node.setub(min(pyo.sqrt(ub_val_node), ub_val_right_node))
        self.val_node.setlb(0)
        self.val_right_node.setlb(max(-pyo.sqrt(ub_val_node), lb_val_right_node))

        raise NotImplementedError()


@declare_custom_block("SqrtOperator")
class SqrtOperatorData(BaseOperatorData):
    _operator_name = "sqrt"
    _is_unary_operator = True

    def build_operator_model(self):
        """Constructs constraint that evaluates the value at the node"""
        self.add_bound_constraints(val_left_node=False)

        # Expressing the constraint in this manner makes it a
        # quadratic constraint, instead of a general nonlinear constraint
        self.evaluate_val_node = Constraint(
            expr=self.val_node * self.val_node == self.val_right_node
        )

    def construct_convex_relaxation(self):
        # Make the domain of the variable non-negative
        ub_val_node = self.val_node.ub
        ub_val_right_node = self.val_right_node.ub

        self.val_node.setub(min(ub_val_node, pyo.sqrt(ub_val_right_node)))
        self.val_right_node.setub(min(ub_val_node**2, ub_val_right_node))
        self.val_right_node.setlb(0)
        self.val_node.setlb(0)

        self.del_component(self.node_val_lb_con)
        self.del_component(self.right_node_val_lb_con)

        raise NotImplementedError()


@declare_custom_block("ExpOperator")
class ExpOperatorData(BaseOperatorData):
    _operator_name = "exp"
    _is_unary_operator = True

    def build_operator_model(self):
        """Constructs constraint that evaluates the value at the node"""
        # Update the domain of variables
        ub_val_node = self.val_node.ub
        ub_val_right_node = self.val_right_node.ub

        self.val_node.setub(min(ub_val_node, pyo.exp(ub_val_right_node)))
        self.val_right_node.setub(min(pyo.log(ub_val_node), ub_val_right_node))
        self.val_node.setlb(0)

        # To avoid numerical issues, we do not let the lower bound of the
        # argument of the exp function go below -10
        if self.val_right_node.lb < -10:
            self.val_right_node.setlb(-10)

        self.add_bound_constraints(val_left_node=False)
        self.del_component(self.node_val_lb_con)

        # Exponential term does not become zero, so multiply it
        # with the operator binary variable to make it zero
        self.evaluate_val_node = Constraint(
            expr=self.val_node == self._bin_var_ref * pyo.exp(self.val_right_node)
        )

    def construct_convex_relaxation(self):
        raise NotImplementedError()


@declare_custom_block("LogOperator")
class LogOperatorData(BaseOperatorData):
    _operator_name = "log"
    _is_unary_operator = True
    _define_aux_var_right = True

    def build_operator_model(self):
        """Constructs constraint that evaluates the value at the node"""
        # Update bounds on variables
        ub_val_node = self.val_node.ub
        ub_val_right_node = self.aux_var_right.ub

        self.val_node.setub(min(ub_val_node, pyo.log(ub_val_right_node)))
        self.aux_var_right.setub(min(pyo.exp(ub_val_node), ub_val_right_node))
        self.val_right_node.setub(self.aux_var_right.ub)

        # Log term need not vanish when the operator is not selected, so
        # multiply it with the operator binary variable to make it zero
        bin_var = self._bin_var_ref
        self.evaluate_val_node = Constraint(
            expr=self.val_node == bin_var * pyo.log(self.aux_var_right)
        )

    def construct_convex_relaxation(self):
        raise NotImplementedError()


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


@declare_custom_block("SampleBlock")
class SampleBlockData(BlockData):
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
                    "op_" + op,
                    OPERATOR_MODELS[op](config={"bin_var": pb.select_operator[n, op]}),
                )
                op_blocks.append(getattr(self.node[n], "op_" + op))

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

        self.square_of_error = pyo.Expression(
            expr=(self.node[1].val_node - output_value) ** 2
        )
