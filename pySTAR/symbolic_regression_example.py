from symbolic_regression import SymbolicRegressionModel
import numpy as np
import pandas as pd
import pyomo.environ as pyo

np.random.seed(42)
data = pd.DataFrame(np.random.uniform(low=1, high=3, size=(10, 2)), columns=["x1", "x2"])
data["y"] = 2.5 * np.exp(data["x1"]) * np.log(data["x2"])
in_cols = ["x1", "x2"]
out_cols = "y"
print(data)

# data = pd.read_csv("3_4_simulation_data_10.csv")
# in_cols = ["liquid_inlet_conc_mol_comp_H2SO4", "solid_inlet_flow_mass"]
# out_cols = "liquid_outlet_flow_vol"


def build_model():
    m = SymbolicRegressionModel(
        data=data,
        input_columns=in_cols,
        output_column=out_cols,
        tree_depth=4,
        unary_operators=["log", "exp"],
        binary_operators=["sum", "diff", "mult"],
        var_bounds=(-100, 100),
        constant_bounds=(-100, 100),
    )

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
    results["sim_data"] = data[out_cols]
    results["prediction"] = [
        mdl.samples[s].node[1].val_node.value
        for s in mdl.samples
    ]
    results["square_of_error"] = [
        mdl.samples[s].square_of_error.expr()
        for s in mdl.samples
    ]

    print(results)
