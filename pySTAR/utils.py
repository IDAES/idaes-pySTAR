from gurobipy import nlfunc
import pyomo.environ as pyo
from bigm_operators import LogOperatorData as BigmLog, ExpOperatorData as BigmExp
from hull_operators import LogOperatorData as HullLog, ExpOperatorData as HullExp
from pySTAR.symbolic_regression import SymbolicRegressionModel


def _bigm_gurobi_formulation(srm: SymbolicRegressionModel):
    for blk in srm.component_data_objects(pyo.Block):
        if isinstance(blk, (BigmLog, BigmExp)):
            # Deactivate the block
            blk.deactivate()

    solver = pyo.SolverFactory("gurobi_persistent")
    solver.set_instance(srm)
    gm = solver._solver_model  # Gurobipy model
    pm_to_gm = solver._pyomo_var_to_solver_var_map
    vlb, vub = srm.var_bounds["lb"], srm.var_bounds["ub"]

    for blk in srm.component_data_objects(pyo.Block):
        if isinstance(blk, BigmLog):
            sb = blk.parent_block()  # Sample block
            val_node = sb.val_node
            op_bin_var = {n: srm.select_operator[n, "log"] for n in srm.nodes_set}

            if vub >= 10:
                _bigm = pyo.value(1e5 - vlb)
            else:
                _bigm = pyo.value(pyo.exp(vub) - vlb)

            gm.addConstr(
                nlfunc.exp(pm_to_gm[val_node[n]]) - pm_to_gm[val_node[2 * n + 1]]
                <= _bigm * (1 - pm_to_gm[op_bin_var[n]])
                for n in srm.non_terminal_nodes_set
            )

    return solver


def _hull_gurobi_formulation(m: SymbolicRegressionModel):
    """Uses Gurobibpy interface to solve the MINLP"""

    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, (HullLog, HullExp)):
            # Deactivate the nonlinear constraint
            blk.evaluate_val_node.deactivate()

    # pylint: disable = protected-access
    grb = pyo.SolverFactory("gurobi_persistent")
    grb.set_instance(m)
    gm = grb._solver_model
    pm_to_gm = grb._pyomo_var_to_solver_var_map

    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, HullLog):
            # Add the nonlinear constraint
            gm.addConstr(
                pm_to_gm[blk.val_node]
                == pm_to_gm[blk.operator_binary]
                * nlfunc.log(pm_to_gm[blk.aux_var_right])
            )

        elif isinstance(blk, HullExp):
            # Add the nonlinear constraint
            gm.addConstr(
                pm_to_gm[blk.val_node]
                == pm_to_gm[blk.operator_binary]
                * nlfunc.exp(pm_to_gm[blk.val_right_node])
            )

    # Solve the optimization model
    grb.solve(tee=True)

    # Activate the constraint back
    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, (HullLog, HullExp)):
            # Deactivate the nonlinear constraint
            blk.evaluate_val_node.activate()

    return grb


def get_gurobi(srm: SymbolicRegressionModel, options: dict | None = None):
    """Returns Gurobi solver object"""
    if options is None:
        # Set default termination criteria
        options = {"MIPGap": 0.01, "TimeLimit": 3600}

    if srm.method_type == "bigm":
        solver = _bigm_gurobi_formulation(srm)
    else:
        solver = _hull_gurobi_formulation(srm)

    solver.options.update(options)
    return solver
