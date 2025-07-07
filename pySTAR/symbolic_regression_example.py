import numpy as np
import pandas as pd
import pyomo.environ as pyo
from symbolic_regression import SymbolicRegressionModel

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
        unary_operators=["square", "sqrt"],
        binary_operators=["sum", "diff", "mult", "div"],
        var_bounds=(-10, 10),
        constant_bounds=(-100, 100),
        model_type="bigm",
    )
    m.add_objective()

    return m


if __name__ == "__main__":
    solver = "baron"
    mdl = build_model()
    mdl.add_redundancy_cuts()
    # mdl.add_tree_size_constraint(3)

    if solver == "scip":
        solver = pyo.SolverFactory("scip")
        solver.solve(mdl, tee=True)

    elif solver == "gurobi":
        mdl.solve_with_gurobi()

    elif solver == "baron":
        solver = pyo.SolverFactory("gams")
        solver.solve(mdl, solver="baron", tee=True)

    print(mdl.get_selected_operators())
    mdl.constant_val.pprint()

    results = pd.DataFrame()
    results["sim_data"] = data["y"]
    results["prediction"] = [mdl.samples[s].val_node[1].value for s in mdl.samples]
    results["square_of_error"] = [
        mdl.samples[s].square_of_residual.expr() for s in mdl.samples
    ]

    print(results)
