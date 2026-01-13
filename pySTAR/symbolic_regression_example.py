import numpy as np
import pandas as pd
import pyomo.environ as pyo
from symbolic_regression import SymbolicRegressionModel
from utils import get_gurobi

np.random.seed(42)
data = pd.DataFrame(
    np.random.uniform(low=1, high=3, size=(10, 2)), columns=["x1", "x2"]
)
data["y"] = data["x1"] ** 2 + data["x2"]
in_cols = ["x1", "x2"]
print(data)


def build_model():
    m = SymbolicRegressionModel(
        data=data,
        input_columns=in_cols,
        output_column="y",
        tree_depth=4,
        operators=["sum", "diff", "mult", "div", "square", "sqrt"],
        var_bounds=(-10, 10),
        constant_bounds=(-100, 100),
        model_type="bigm",
    )
    m.add_objective()

    return m


if __name__ == "__main__":
    solver = "baron"
    mdl = build_model()
    # mdl.add_tree_size_constraint(3)

    if solver == "scip":
        solver = pyo.SolverFactory("scip")
        solver.solve(mdl, tee=True)

    elif solver == "gurobi":
        solver = get_gurobi(mdl)
        solver.solve(tee=True)

    elif solver == "baron":
        solver = pyo.SolverFactory("gams")
        solver.solve(mdl, solver="baron", tee=True)

    et = mdl.selected_tree_to_expression()
    print("Expression: ", et)
    print("R2 value ", mdl.compute_r2())
    print("Constant values: ", et.cst_values)
