import numpy as np
import pandas as pd
import pyomo.environ as pyo

from pystar.core.symbolic_regression import SymbolicRegressionModel

# Experiment settings
num_samples = 3
num_test_samples = 1000
tree_depth = 2
time_limit = 600
cplex_lib_name = r"C:\GAMS\48\cplex2211.dll"
operators = ["mult", "div", "square", "sqrt", "exp", "log"]


# Generate signed data for: y = x1*x2 + 0.2*x2
rng = np.random.default_rng(42)
data = pd.DataFrame(
    {
        "x1": rng.uniform(1.0, 2.0, num_samples),
        "x2": rng.uniform(1.0, 2.0, num_samples),
    }
)

# points_per_var = 4

# x1_grid = np.linspace(-1.0, 1.0, points_per_var)
# x2_grid = np.linspace(-1.0, 1.0, points_per_var)

# x1, x2 = np.meshgrid(x1_grid, x2_grid)

# data = pd.DataFrame(
#    {
#        "x1": x1.ravel(),
#        "x2": x2.ravel(),
#    }
# )

data["y"] = data["x1"] * data["x2"]

print(data)


m = SymbolicRegressionModel(
    data=data,
    input_columns=["x1", "x2"],
    output_column="y",
    tree_depth=tree_depth,
    operators=operators,
    var_bounds=(-1, 1),
    constant_bounds=(0, 5.0),
    model_type="hull",
)

m.add_objective("sse")


def inspect_operator_block(blk, ostream=None):
    print(f"\n--- {blk.name} ---", file=ostream)

    print("\nActive constraints:", file=ostream)
    for con in blk.component_data_objects(
        pyo.Constraint, active=True, descend_into=False
    ):
        print(f"{con.name}: {con.expr}", file=ostream)

    print("\nInactive constraints:", file=ostream)
    for con in blk.component_data_objects(
        pyo.Constraint, active=False, descend_into=False
    ):
        print(f"{con.name}: {con.expr}", file=ostream)


blk = m.samples[0].node[1].log_operator
blk.construct_convex_relaxation()

with open("log.txt", "w") as f:
    inspect_operator_block(blk, ostream=f)
