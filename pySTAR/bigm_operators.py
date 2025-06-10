"""
This module contains all the supported operator models
for symbolic regression. The operator constraints are implemented using
the big-m approach
"""

import pyomo.environ as pyo
from pySTAR.custom_block.custom_block import BlockData, declare_custom_block


@declare_custom_block("BaseOperator")
class BaseOperatorData(BlockData):
    """Operator template for big-m type constraints"""

    @property
    def symbolic_regression_model(self):
        """Returns a pointer to the symbolic regression model"""
        # Model layout: SymbolicRegressionModel
        #                 |___SampleBlock
        #                        |___OperatorBlock
        # Regression model is two levels up, so calling the
        # parent_block method twice. The model is needed to access the
        # operator binary variables.
        return self.parent_block().parent_block()

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        """
        Builds the operator-specific constraints

        Parameters
        ----------
        val_node : Var
            Variables corresponding to the value at a node. It is an
            indexed Var, where the index denotes the node number.

        op_bin_var : dict
            Dictionary containing the binary variables associated with
            the operator. Here, the keys correspond to node numbers, and the
            values contain the operator binary variable.
        """
        raise NotImplementedError("Operator-specific constraints are not implemented")

    def construct_convex_relaxation(self):
        """Constructs a convex relaxation of the operator-specific constraints"""
        raise NotImplementedError()


@declare_custom_block("SumOperator")
class SumOperatorData(BaseOperatorData):
    """Adds constraints for the addition operator"""

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = vub - 2 * vlb
            return val_node[n] - (val_node[2 * n] + val_node[2 * n + 1]) <= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = vlb - 2 * vub
            return val_node[n] - (val_node[2 * n] + val_node[2 * n + 1]) >= bigm * (
                1 - op_bin_var[n]
            )

    def construct_convex_relaxation(self):
        # Constraints are convex, so returning the block as it is
        pass


@declare_custom_block("DiffOperator")
class DiffOperatorData(BaseOperatorData):
    """Template for difference operator"""

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        """Function containing operator-specific constraints"""
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = 2 * vub - vlb
            return val_node[n] - (val_node[2 * n] - val_node[2 * n + 1]) <= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = 2 * vlb - vub
            return val_node[n] - (val_node[2 * n] - val_node[2 * n + 1]) >= bigm * (
                1 - op_bin_var[n]
            )

    def construct_convex_relaxation(self):
        # Constraints are convex, so returning the block as it is
        pass


@declare_custom_block("MultOperator")
class MultOperatorData(BaseOperatorData):
    """Template for difference operator"""

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        """Function containing operator-specific constraints"""
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]
        product_values = [
            pyo.value(vlb * vlb),
            pyo.value(vlb * vub),
            pyo.value(vub * vub),
        ]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = vub - min(product_values)
            return val_node[n] - (val_node[2 * n] * val_node[2 * n + 1]) <= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = vlb - max(product_values)
            return val_node[n] - (val_node[2 * n] * val_node[2 * n + 1]) >= bigm * (
                1 - op_bin_var[n]
            )

    def construct_convex_relaxation(self):
        raise NotImplementedError()


@declare_custom_block("DivOperator")
class DivOperatorData(BaseOperatorData):
    """Template for difference operator"""

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        """Function containing operator-specific constraints"""
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]
        eps = srm.eps_value
        product_values = [
            pyo.value(vlb * vlb),
            pyo.value(vlb * vub),
            pyo.value(vub * vub),
        ]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = max(product_values) - vlb
            return (val_node[n] * val_node[2 * n + 1]) - val_node[2 * n] <= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = min(product_values) - vub
            return (val_node[n] * val_node[2 * n + 1]) - val_node[2 * n] >= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_modes_set)
        def avoid_zero_constraint(_, n):
            # Ensures that the denominator is away from zero
            # return eps * op_bin_var[n] <= val_node[2 * n + 1] ** 2
            return op_bin_var[n] * val_node[2 * n + 1] ** 2 >= op_bin_var[n] - 1 + eps

    def construct_convex_relaxation(self):
        raise NotImplementedError()


@declare_custom_block("SqrtOperator")
class SqrtOperatorData(BaseOperatorData):
    """Template for difference operator"""

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        """Function containing operator-specific constraints"""
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = max(vlb**2, vub**2) - vlb
            return val_node[n] ** 2 - val_node[2 * n + 1] <= bigm * (1 - op_bin_var[n])

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = -vub
            return val_node[n] ** 2 - val_node[2 * n + 1] >= bigm * (1 - op_bin_var[n])

        @self.Constraint(srm.non_terminal_modes_set)
        def non_negativity_constraint(_, n):
            # Ensures that the denominator is away from zero
            # return eps * op_bin_var[n] <= val_node[2 * n + 1] ** 2
            return op_bin_var[n] * val_node[2 * n + 1] >= 0

    def construct_convex_relaxation(self):
        raise NotImplementedError()


@declare_custom_block("ExpOperator")
class ExpOperatorData(BaseOperatorData):
    """Template for difference operator"""

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        """Function containing operator-specific constraints"""
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = vub
            return val_node[n] - pyo.exp(val_node[2 * n + 1]) <= bigm * (
                1 - op_bin_var[n]
            )

        # NOTE: Would this constrain the value of v[2*n+1]?
        if vub >= 10:
            _bigm = vlb - 1e5
        else:
            _bigm = vlb - pyo.exp(vub)

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            return val_node[n] - pyo.exp(val_node[2 * n + 1]) >= _bigm * (
                1 - op_bin_var[n]
            )

    def construct_convex_relaxation(self):
        raise NotImplementedError()


@declare_custom_block("LogOperator")
class LogOperatorData(BaseOperatorData):
    """Template for difference operator"""

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        """Function containing operator-specific constraints"""
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]
        eps = srm.eps_value

        # NOTE: Would this constrain the value of v[2*n+1]?
        # Not certain about this logic. Need to doublecheck.
        if vub - vlb >= 10:
            _bigm = 1e5
        elif pyo.exp(vub - vlb) < -1e-5:
            _bigm = -1e5
        else:
            _bigm = pyo.exp(vub - vlb)

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            return pyo.exp(val_node[n]) - val_node[2 * n + 1] <= _bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = -vub
            return pyo.exp(val_node[n]) - val_node[2 * n + 1] >= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_modes_set)
        def avoid_zero_constraint(_, n):
            # Ensures that the denominator is away from zero
            # return eps * op_bin_var[n] <= val_node[2 * n + 1] ** 2
            return op_bin_var[n] * val_node[2 * n + 1] >= op_bin_var[n] - 1 + eps

    def construct_convex_relaxation(self):
        raise NotImplementedError()


@declare_custom_block("SquareOperator")
class SquareOperatorData(BaseOperatorData):
    """Template for difference operator"""

    def build(self, val_node: pyo.Var, op_bin_var: dict):
        """Function containing operator-specific constraints"""
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = vub
            return val_node[n] - val_node[2 * n + 1] ** 2 <= bigm * (1 - op_bin_var[n])

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = vlb - max(vlb**2, vub**2)
            return val_node[n] - val_node[2 * n + 1] ** 2 >= bigm * (1 - op_bin_var[n])

    def construct_convex_relaxation(self):
        raise NotImplementedError()


# pylint: disable = undefined-variable
BIGM_OPERATORS = {
    "sum": SumOperator,
    "diff": DiffOperator,
    "mult": MultOperator,
    "div": DivOperator,
    "sqrt": SqrtOperator,
    "log": LogOperator,
    "exp": ExpOperator,
    "square": SquareOperator,
}


@declare_custom_block("BigmSampleBlock")
class BigmSampleBlockData(BlockData):
    """Class for evaluating the expression tree for each sample"""

    @property
    def symbolic_regression_model(self):
        """Returns a pointer to the symbolic regression model"""
        return self.parent_block()

    # pylint: disable=attribute-defined-outside-init
    def build(self):
        """Builds the expression tree for a given sample"""
        srm = self.symbolic_regression_model
        self.val_node = pyo.Var(
            srm.nodes_set,
            bounds=(srm.var_bounds["lb"], srm.var_bounds["ub"]),
            doc="Value at each node",
        )

        for op in srm.operators_set:
            # Get binary variables corresponding to the opeartor
            # and save them in a dictionary. Here, the keys are node
            # numbers and the values are Var objects.
            op_bin_vars = dict(srm.select_operator[:, op].wildcard_items())
            setattr(
                self,
                op + "_operator",
                BIGM_OPERATORS[op](
                    model_options={
                        "val_node": self.val_node,
                        "op_bin_var": op_bin_vars,
                    }
                ),
            )
