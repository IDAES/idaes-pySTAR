from gurobipy import nlfunc
import pyomo.environ as pyo
import bigm_operators as bop
import hull_operators as hop
from symbolic_regression import SymbolicRegressionModel


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

    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, (hop.ExpOperatorData, hop.LogOperatorData)):
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

    # Activate the constraint back
    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, (hop.LogOperatorData, hop.ExpOperatorData)):
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
