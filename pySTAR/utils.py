from gurobipy import nlfunc
import pyomo.environ as pyo
from pySTAR.operators.operators import (
    LogOperatorData,
    ExpOperatorData,
)


def solve_with_gurobi(m):
    """Uses Gurobibpy interface to solve the MINLP"""

    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, (LogOperatorData, ExpOperatorData)):
            # Deactivate the nonlinear constraint
            blk.evaluate_val_node.deactivate()

    # pylint: disable = protected-access
    grb = pyo.SolverFactory("gurobi_persistent")
    grb.set_instance(m)
    gm = grb._solver_model
    pm_to_gm = grb._pyomo_var_to_solver_var_map

    for blk in m.component_data_objects(pyo.Block):
        if isinstance(blk, LogOperatorData):
            # Add the nonlinear constraint
            gm.addConstr(
                pm_to_gm[blk.val_node]
                == pm_to_gm[blk.operator_binary]
                * nlfunc.log(pm_to_gm[blk.aux_var_right])
            )

        elif isinstance(blk, ExpOperatorData):
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
        if isinstance(blk, (LogOperatorData, ExpOperatorData)):
            # Deactivate the nonlinear constraint
            blk.evaluate_val_node.activate()
