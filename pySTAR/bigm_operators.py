"""
This module contains all the supported operator models
for symbolic regression. The operator constraints are implemented using
the big-m approach
"""

import pandas as pd
from pyomo.core.base.block import BlockData, declare_custom_block
import pyomo.environ as pyo


@declare_custom_block("BaseOperator", rule="build")
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

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
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

    @staticmethod
    def compute_node_value(left_child: float, right_child: float):
        """Computes the value at a node based on the operator"""
        raise NotImplementedError()


@declare_custom_block("SumOperator", rule="build")
class SumOperatorData(BaseOperatorData):
    """Adds constraints for the addition operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
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

    @staticmethod
    def compute_node_value(left_child, right_child):
        return left_child + right_child


@declare_custom_block("DiffOperator", rule="build")
class DiffOperatorData(BaseOperatorData):
    """Adds constraints for the difference operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
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

    @staticmethod
    def compute_node_value(left_child, right_child):
        return left_child - right_child


@declare_custom_block("MultOperator", rule="build")
class MultOperatorData(BaseOperatorData):
    """Adds constraints for the multiplication operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
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

    @staticmethod
    def compute_node_value(left_child, right_child):
        return left_child * right_child


@declare_custom_block("DivOperator", rule="build")
class DivOperatorData(BaseOperatorData):
    """Adds constraints for the division operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
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

        @self.Constraint(srm.non_terminal_nodes_set)
        def avoid_zero_constraint(_, n):
            # Ensures that the denominator is away from zero
            # return eps * op_bin_var[n] <= val_node[2 * n + 1] ** 2
            return val_node[2 * n + 1] ** 2 >= op_bin_var[n] - 1 + eps

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(left_child, right_child):
        return left_child / right_child


@declare_custom_block("SqrtOperator", rule="build")
class SqrtOperatorData(BaseOperatorData):
    """Adds constraints for the square root operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = max(pyo.value(vlb * vlb), pyo.value(vub * vub)) - vlb
            return val_node[n] ** 2 - val_node[2 * n + 1] <= bigm * (1 - op_bin_var[n])

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = -vub
            return val_node[n] ** 2 - val_node[2 * n + 1] >= bigm * (1 - op_bin_var[n])

        @self.Constraint(srm.non_terminal_nodes_set)
        def non_negativity_constraint(_, n):
            return val_node[2 * n + 1] >= (1 - op_bin_var[n]) * vlb

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.sqrt(right_child)


@declare_custom_block("ExpOldOperator", rule="build")
class ExpOldOperatorData(BaseOperatorData):
    """Adds constraints for the exponential operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
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

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.exp(right_child)


@declare_custom_block("ExpCompOperator", rule="build")
class ExpCompOperatorData(BaseOperatorData):
    """Adds constraints for the exponential operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
        srm = self.symbolic_regression_model

        @self.Constraint(srm.non_terminal_nodes_set)
        def complementarity_constraint(_, n):
            return op_bin_var[n] * (val_node[n] - pyo.exp(val_node[2 * n + 1])) == 0

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.exp(right_child)


@declare_custom_block("ExpOperator", rule="build")
class ExpOperatorData(BaseOperatorData):
    """Adds constraints for the exponential operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        # pylint: disable = attribute-defined-outside-init
        self.aux_var_exp = pyo.Var(
            srm.non_terminal_nodes_set,
            bounds=(max(-10, vlb), pyo.log(vub)),
            doc="Auxiliary variable for modeling exp function",
        )

        @self.Constraint(srm.non_terminal_nodes_set)
        def func_upper_bound_constraint(blk, n):
            bigm = vub - 0
            return val_node[n] - pyo.exp(blk.aux_var_exp[n]) <= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def func_lower_bound_constraint(blk, n):
            bigm = vlb - vub
            return val_node[n] - pyo.exp(blk.aux_var_exp[n]) >= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def var_upper_bound_constraint(blk, n):
            bigm = vub - blk.aux_var_exp[n].lb
            return val_node[2 * n + 1] - blk.aux_var_exp[n] <= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def var_lower_bound_constraint(blk, n):
            bigm = vlb - blk.aux_var_exp[n].ub
            return val_node[2 * n + 1] - blk.aux_var_exp[n] >= bigm * (
                1 - op_bin_var[n]
            )

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.exp(right_child)


@declare_custom_block("LogOldOperator", rule="build")
class LogOldOperatorData(BaseOperatorData):
    """Adds constraints for the logarithm operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]
        eps = srm.eps_value

        # NOTE: Would this constrain the value of v[2*n+1]?
        if vub >= 10:
            _bigm = 1e5 - vlb
        else:
            _bigm = pyo.exp(vub) - vlb

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

        @self.Constraint(srm.non_terminal_nodes_set)
        def avoid_zero_constraint(_, n):
            # Ensures that the denominator is away from zero
            # return eps * op_bin_var[n] <= val_node[2 * n + 1] ** 2
            return op_bin_var[n] * val_node[2 * n + 1] >= op_bin_var[n] - 1 + eps

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.log(right_child)


@declare_custom_block("LogCompOperator", rule="build")
class LogCompOperatorData(BaseOperatorData):
    """Adds constraints for the exponential operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
        srm = self.symbolic_regression_model

        @self.Constraint(srm.non_terminal_nodes_set)
        def complementarity_constraint(_, n):
            # Need to transform the equation to avoid log domain error
            return op_bin_var[n] * (pyo.exp(val_node[n]) - val_node[2 * n + 1]) == 0

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.log(right_child)


@declare_custom_block("LogOperator", rule="build")
class LogOperatorData(BaseOperatorData):
    """Adds constraints for the exponential operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        # pylint: disable = attribute-defined-outside-init
        self.aux_var_log = pyo.Var(
            srm.non_terminal_nodes_set,
            bounds=(srm.eps_value, vub),
            doc="Auxiliary variable for modeling log function",
        )

        @self.Constraint(srm.non_terminal_nodes_set)
        def func_upper_bound_constraint(blk, n):
            bigm = vub - pyo.log(blk.aux_var_log[n].lb)
            return val_node[n] - pyo.log(blk.aux_var_log[n]) <= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def func_lower_bound_constraint(blk, n):
            bigm = vlb - pyo.log(blk.aux_var_log[n].ub)
            return val_node[n] - pyo.log(blk.aux_var_log[n]) >= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def var_upper_bound_constraint(blk, n):
            bigm = vub - blk.aux_var_log[n].lb
            return val_node[2 * n + 1] - blk.aux_var_log[n] <= bigm * (
                1 - op_bin_var[n]
            )

        @self.Constraint(srm.non_terminal_nodes_set)
        def var_lower_bound_constraint(blk, n):
            bigm = vlb - blk.aux_var_log[n].ub
            return val_node[2 * n + 1] - blk.aux_var_log[n] >= bigm * (
                1 - op_bin_var[n]
            )

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return pyo.log(right_child)


@declare_custom_block("SquareOperator", rule="build")
class SquareOperatorData(BaseOperatorData):
    """Adds constraints for the square operator"""

    def build(self, *args, val_node: pyo.Var, op_bin_var: dict):
        srm = self.symbolic_regression_model
        vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

        @self.Constraint(srm.non_terminal_nodes_set)
        def upper_bound_constraint(_, n):
            bigm = vub
            return val_node[n] - val_node[2 * n + 1] ** 2 <= bigm * (1 - op_bin_var[n])

        @self.Constraint(srm.non_terminal_nodes_set)
        def lower_bound_constraint(_, n):
            bigm = vlb - max(pyo.value(vlb * vlb), pyo.value(vub * vub))
            return val_node[n] - val_node[2 * n + 1] ** 2 >= bigm * (1 - op_bin_var[n])

    def construct_convex_relaxation(self):
        raise NotImplementedError()

    @staticmethod
    def compute_node_value(_, right_child):
        return right_child**2


# pylint: disable = undefined-variable
BIGM_OPERATORS = {
    "sum": SumOperator,
    "diff": DiffOperator,
    "mult": MultOperator,
    "div": DivOperator,
    "sqrt": SqrtOperator,
    "log": LogOperator,
    "log_comp": LogCompOperator,
    "log_old": LogOldOperator,
    "exp": ExpOperator,
    "exp_comp": ExpCompOperator,
    "exp_old": ExpOldOperator,
    "square": SquareOperator,
}


@declare_custom_block("BigmSampleBlock", rule="build")
class BigmSampleBlockData(BlockData):
    """Class for evaluating the expression tree for each sample"""

    @property
    def symbolic_regression_model(self):
        """Returns a pointer to the symbolic regression model"""
        return self.parent_block()

    # pylint: disable=attribute-defined-outside-init
    def build(self, *args):
        """Builds the expression tree for a given sample"""
        srm = self.symbolic_regression_model
        self.val_node = pyo.Var(
            srm.nodes_set,
            bounds=(srm.var_bounds["lb"], srm.var_bounds["ub"]),
            doc="Value at each node",
        )

        # Value defining constraints for operands
        data = srm.input_data_ref.loc[args[0]]

        @self.Constraint(srm.nodes_set)
        def value_upper_bound_constraint(blk, n):
            return blk.val_node[n] <= srm.constant_val[n] + sum(
                data[op] * srm.select_operator[n, op]
                for op in srm.operands_set
                if op != "cst"
            ) + srm.var_bounds["ub"] * sum(
                srm.select_operator[n, op] for op in srm.operators_set
            )

        @self.Constraint(srm.nodes_set)
        def value_lower_bound_constraint(blk, n):
            return blk.val_node[n] >= srm.constant_val[n] + sum(
                data[op] * srm.select_operator[n, op]
                for op in srm.operands_set
                if op != "cst"
            ) + srm.var_bounds["lb"] * sum(
                srm.select_operator[n, op] for op in srm.operators_set
            )

        for op in srm.operators_set:
            # Get binary variables corresponding to the opeartor
            # and save them in a dictionary. Here, the keys are node
            # numbers and the values are Var objects.
            op_bin_vars = {n: srm.select_operator[n, op] for n in srm.nodes_set}
            setattr(
                self,
                op + "_operator",
                BIGM_OPERATORS[op](val_node=self.val_node, op_bin_var=op_bin_vars),
            )

        self.residual = pyo.Var(doc="Residual value for the sample")
        self.calculate_residual = pyo.Constraint(
            expr=self.residual == srm.output_data_ref[self.index()] - self.val_node[1],
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
            return (blk.val_node[2 * n] - blk.val_node[2 * n + 1]) >= (vlb - vub) * (
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
                    true_value[n] += getattr(self, op + "_operator").compute_node_value(
                        true_value[2 * n], true_value[2 * n + 1]
                    )

        computed_value = {n: self.val_node[n].value for n in self.val_node}

        return pd.DataFrame.from_dict(
            {"True Value": true_value, "Computed Value": computed_value}
        )
