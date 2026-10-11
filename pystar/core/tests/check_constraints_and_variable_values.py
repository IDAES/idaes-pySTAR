import numpy as np
import pandas as pd
import pyomo.environ as pyo

from pystar.core.symbolic_regression import SymbolicRegressionModel
from pystar.core.utils import add_all_constraints

# Experiment settings
num_samples = 10
tree_depth = 2
operators = ["exp", "log"]


# Generate signed data for: y = x1*x2
rng = np.random.default_rng(42)
data = pd.DataFrame(
    {
        "x1": rng.uniform(1.0, 2.0, num_samples),
        "x2": rng.uniform(1.0, 2.0, num_samples),
    }
)
data["y"] = data["x1"] * data["x2"]
print(data)

m = SymbolicRegressionModel(
    data=data,
    input_columns=["x1", "x2"],
    output_column="y",
    tree_depth=tree_depth,
    operators=operators,
    var_bounds=(-10, 100),
    constant_bounds=(-10, -1),
    model_type="bigm",
)

m.add_objective("sse")
#m.constrain_min_tree_size(2)
#m = add_all_constraints(m)
#m.relax_nonconvex_constraints()

#m.add_similar_operation_cuts()
#m.add_associative_operation_cuts()
#m.add_constant_operation_cuts()
#m.add_implication_cuts()
#m.add_inverse_function_composition_cuts()
#m.add_same_operand_operation_cuts()

#m.add_symmetry_breaking_cuts()

#m.select_operator[1, "exp"].fix(1)
#m.samples[1].log_operator.aux_var_log[1].fix(0.02)
#m.samples[0].exp_operator.aux_var_exp[1].fix(1)
#m.samples[0].exp_operator.aux_var_exp[1].fix(data.loc[0, "x1"])

with open("Constraints.txt", "w") as f:
    for con in m.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        f.write(f"{con.name}: {con.expr}\n")

with open("Full_pyomo_model.txt", "w") as f:
    m.pprint(ostream=f) #output stream is the file f


solver = pyo.SolverFactory("baron")

solver.options["MaxTime"] = 100
#solver.options["NumSol"] = 3
#solver.options["CompIIS"] = 1
solver.options["CplexLibName"] = r"C:\baron\cplex2212.dll"

results = solver.solve(
    m,
    tee=True,
    symbolic_solver_labels=True,
    keepfiles=True,
    solnfile=f"res_MILP.sol",
    logfile=f"log_MILP.log",
)

print("SR model:", m.selected_tree_to_expression())
print("Selected operators:", m.get_selected_operators())

print("Objective value:", pyo.value(m.sse))

# Only (one) optimal solution stored in the pyomo model can be saved 
previous_solution = {}
for var in m.component_data_objects(pyo.Var): # contains variables m.x[1], m.x[2] etc
    if var.value is not None:
        previous_solution[var.name] = var.value
print(
    f"Saved {len(previous_solution)} variable values for warm-start in next optimization."
)
#print("Variable values:", previous_solution)

with open("Optimal_variable_values.txt", "w") as f:
    for name, value in previous_solution.items():
        print(f"{name} = {value}", file=f)