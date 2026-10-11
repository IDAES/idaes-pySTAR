import numpy as np
import pandas as pd
import pyomo.environ as pyo

from pystar.core.symbolic_regression import SymbolicRegressionModel

num_samples = 1
tree_depth = 3
operators = ["sum", "diff", "mult", "div", "sqrt", "log", "square", "exp"] #, "square", "sum", "diff", "exp", "mult", "div", "sqrt", "log"]

# Generate signed data for: y = x1*x2
rng = np.random.default_rng(42)
data = pd.DataFrame(
    {
        "x1": rng.uniform(1.0, 2.0, num_samples),
        "x2": rng.uniform(1.0, 2.0, num_samples),
    }
)

data["y"] = data["x1"] * data["x2"]
#print(data)

input_columns = ["x1", "x2"]

m = SymbolicRegressionModel(
    data=data,
    input_columns=input_columns,
    output_column="y",
    tree_depth=tree_depth,
    operators=operators,
    var_bounds=(-10, 10),
    constant_bounds=(-10, 10),
    model_type="hull",
)

m.add_objective("sse")
for add_cuts in (
    m.add_associative_operation_cuts,
    m.add_constant_operation_cuts,
    m.add_implication_cuts,
    m.add_inverse_function_composition_cuts,
    m.add_same_operand_operation_cuts,
    m.add_similar_operation_cuts,
    m.add_symmetry_breaking_cuts,
):
    # Include nested sample blocks, where symmetry-breaking cuts are added.
    existing_constraints = {
        id(constraint)
        for constraint in m.component_data_objects(pyo.Constraint, descend_into=True)
    }
    add_cuts()
    added_constraints = [
        constraint
        for constraint in m.component_data_objects(pyo.Constraint, descend_into=True)
        if id(constraint) not in existing_constraints
    ]

    print(f"\n{add_cuts.__name__}(): {len(added_constraints)} constraints added")
    for constraint in added_constraints:
        print(f"{constraint.name}: {constraint.expr}")
