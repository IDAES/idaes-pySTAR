import numpy as np
import pandas as pd
import pyomo.environ as pyo

from pystar.core.symbolic_regression import SymbolicRegressionModel
from pystar.solution_approach.MIQP_NLPs_algorithm import MIQP_NLPs_algorithm

# User-defined
num_samples = 10
tree_depth = 2
operators = ["sum", "div", "mult"] #, "diff", "square", "sqrt", "exp", "log"]

# Generate data for: y = x1*x2
rng = np.random.default_rng(42)
data = pd.DataFrame(
    {
        "x1": rng.uniform(1.0, 2.0, num_samples),
        "x2": rng.uniform(1.0, 2.0, num_samples),
    }
)
data["y"] = data["x1"] * data["x2"]
#data["y"]=data["y_clean"] + rng.normal(loc=0, scale=0.1, size=num_samples)


m = SymbolicRegressionModel(
    data=data,
    input_columns=["x1", "x2"],
    output_column="y",
    tree_depth=tree_depth,
    operators=operators,
    var_bounds=(-10, 10),
    constant_bounds=(-10, 10),
    model_type="hull",
)

m.add_objective("sse")

m.add_associative_operation_cuts()
m.add_constant_operation_cuts()
m.add_implication_cuts()
m.add_inverse_function_composition_cuts()
m.add_same_operand_operation_cuts()
m.add_similar_operation_cuts()

#m.add_symmetry_breaking_cuts()


optimal_objective, optimal_solution_index, best_model, expression_trees, shrank_pool, global_optimality, info = MIQP_NLPs_algorithm(m, k=200, time_limit=100)

print("\nAll expression trees:", expression_trees)
#print("\nShrank pool of NLP objectives:", shrank_pool)

print("\nGlobally optimal solution")
print("Index (0-based) of globally optimal solution in shrank pool:", optimal_solution_index)
print("Globally optimal objective:", optimal_objective)
print("Selected operators", best_model.get_selected_operators())
print("Globally optimal model:", best_model.selected_tree_to_expression())