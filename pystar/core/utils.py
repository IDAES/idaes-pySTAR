from gurobipy import nlfunc
import pyomo.environ as pyo
import pystar.core.bigm_operators as bop
import pystar.core.hull_operators as hop
from pystar.core.symbolic_regression import SymbolicRegressionModel


def _bigm_gurobi_formulation(srm: SymbolicRegressionModel):
    for blk in srm.component_data_objects(pyo.Block):
        if isinstance(blk, (bop.ExpOperatorData, bop.LogOperatorData)):
            # Deactivate the nonlinear constraints
            blk.func_upper_bound_constraint.deactivate()
            blk.func_lower_bound_constraint.deactivate()

    solver = pyo.SolverFactory("gurobi_persistent")
    solver.set_instance(srm)
    gm = solver._solver_model  # Gurobipy model
    pm_to_gm = solver._pyomo_var_to_solver_var_map
    vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

    for blk in srm.component_data_objects(pyo.Block):
        if isinstance(blk, bop.LogOperatorData):
            sb = blk.parent_block()  # Sample block
            val_node = sb.val_node
            op_bin_var = {n: srm.select_operator[n, "log"] for n in srm.nodes_set}

            aux_vars = gm.addVars(list(srm.non_terminal_nodes_set))
            gm.addConstrs(
                aux_vars[n] == nlfunc.log(pm_to_gm[blk.aux_var_log[n]])
                for n in srm.non_terminal_nodes_set
            )
            gm.addConstrs(
                pm_to_gm[val_node[n]] - aux_vars[n]
                <= (vub - pyo.log(blk.aux_var_log[n].lb))
                * (1 - pm_to_gm[op_bin_var[n]])
                for n in srm.non_terminal_nodes_set
            )
            gm.addConstrs(
                pm_to_gm[val_node[n]] - aux_vars[n]
                >= (vlb - pyo.log(blk.aux_var_log[n].ub))
                * (1 - pm_to_gm[op_bin_var[n]])
                for n in srm.non_terminal_nodes_set
            )

        if isinstance(blk, bop.ExpOperatorData):
            sb = blk.parent_block()  # Sample block
            val_node = sb.val_node
            op_bin_var = {n: srm.select_operator[n, "exp"] for n in srm.nodes_set}

            aux_vars = gm.addVars(list(srm.non_terminal_nodes_set))
            gm.addConstrs(
                aux_vars[n] == nlfunc.exp(pm_to_gm[blk.aux_var_exp[n]])
                for n in srm.non_terminal_nodes_set
            )
            gm.addConstrs(
                pm_to_gm[val_node[n]] - aux_vars[n]
                <= (vub - 0) * (1 - pm_to_gm[op_bin_var[n]])
                for n in srm.non_terminal_nodes_set
            )
            gm.addConstrs(
                pm_to_gm[val_node[n]] - aux_vars[n]
                >= (vlb - vub) * (1 - pm_to_gm[op_bin_var[n]])
                for n in srm.non_terminal_nodes_set
            )

    for blk in srm.component_data_objects(pyo.Block):
        if isinstance(blk, (bop.ExpOperatorData, bop.LogOperatorData)):
            # Activate the nonlinear constraints
            blk.func_upper_bound_constraint.activate()
            blk.func_lower_bound_constraint.activate()

    return solver


def _hull_gurobi_formulation(m: SymbolicRegressionModel):
    """Uses Gurobibpy interface to solve the MINLP"""
    op_list = (
        hop.ExpOperatorData,
        hop.LogOperatorData,
        hop.SqrtOperatorData,
    )

    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, op_list):
            # Deactivate the nonlinear constraint
            blk.evaluate_val_node.deactivate()

    # pylint: disable = protected-access
    grb = pyo.SolverFactory("gurobi_persistent")
    grb.set_instance(m)
    gm = grb._solver_model
    pm_to_gm = grb._pyomo_var_to_solver_var_map

    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, hop.LogOperatorData):
            # Add the nonlinear constraint
            gm.addConstr(
                pm_to_gm[blk.val_node] == nlfunc.log(pm_to_gm[blk.aux_var_right])
            )

        elif isinstance(blk, hop.ExpOperatorData):
            # Add the nonlinear constraint
            gm.addConstr(
                pm_to_gm[blk.val_node]
                == nlfunc.exp(pm_to_gm[blk.val_right_node])
                + pm_to_gm[blk.operator_binary]
                - 1
            )

        elif isinstance(blk, hop.SqrtOperatorData):
            # Add the nonlinear constraint
            gm.addConstr(
                pm_to_gm[blk.val_node] == nlfunc.sqrt(pm_to_gm[blk.val_right_node])
            )

    # Activate the constraint back
    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, op_list):
            # Activate the nonlinear constraint
            blk.evaluate_val_node.activate()

    return grb


def get_gurobi(srm: SymbolicRegressionModel, options: dict | None = None):
    """Returns Gurobi solver object"""
    if options is None:
        # Set default termination criteria
        options = {"MIPGap": 0.01, "TimeLimit": 3600}

    if srm.model_type == "bigm":
        solver = _bigm_gurobi_formulation(srm)
    else:
        solver = _hull_gurobi_formulation(srm)

    solver.options.update(options)
    return solver

def only_tree_structure(srm: SymbolicRegressionModel):
    """SR model with tree-structure constraints only.
    That is, value-defining constraints are excluded."""

    srm.samples.deactivate()
    srm.constant_lb_con.deactivate()
    srm.constant_ub_con.deactivate()

    return srm

def no_right_cst_constraints(srm: SymbolicRegressionModel):
    @srm.Constraint(srm.nodes_set)
    def no_right_cst_con(blk, n):
        if n % 2 == 1 and n > 1:
            return blk.select_operator[n, "cst"] == 0
        else:
            return pyo.Constraint.Skip

    return srm

def weak_add_constant_operation_cuts_1(srm: SymbolicRegressionModel, use_unit_bound=True):
    "Category 1 eliminates: cst +- (cst +- A), and: cst */ (cst */ A)."

    @srm.Constraint(srm.pre_non_terminal_nodes_set, srm.binary_op_pairs_set)
    def weak_redundant_cst_operations_1(blk, n, op1, op2):

        # RHS is either 1 or delta_n
        rhs = 1 if use_unit_bound else blk.select_node[n]

        return (
            blk.select_operator[2 * n, "cst"]
            + blk.select_operator[4 * n + 2, "cst"]
            <= 3 * rhs
            - blk.select_operator[n, op1]
            - blk.select_operator[2 * n + 1, op2]
        )
    
    return srm

def weak_add_constant_operation_cuts_2(srm: SymbolicRegressionModel, use_unit_bound=True):
    "Category 2 eliminates: (cst */ A) */ (cst */ B) and (cst +- A) +- (cst +- B)."

    @srm.Constraint(srm.pre_non_terminal_nodes_set, srm.same_family_triples_set)
    def weak_redundant_cst_operations_2(blk, n, op1, op2, op3):

        # RHS is either 1 or delta_n
        rhs = 1 if use_unit_bound else blk.select_node[n]

        return (
            blk.select_operator[4 * n + 2, "cst"]
            + blk.select_operator[2 * n, "cst"]
            <= 4 * rhs 
            - blk.select_operator[n, op1]
            - blk.select_operator[2 * n, op2]
            - blk.select_operator[2 * n + 1, op3]
        )

    return srm

def remove_category_cst_manipulation_cut(srm: SymbolicRegressionModel, remove_category: int):
    """Remove certain category of redundant constant manipulation cuts from the model."""

    if remove_category == 1:
        srm.redundant_cst_operations_1.deactivate()
    elif remove_category == 2:
        srm.redundant_cst_operations_2.deactivate()
    elif remove_category == 3:
        srm.redundant_cst_operations_3.deactivate()

    return srm