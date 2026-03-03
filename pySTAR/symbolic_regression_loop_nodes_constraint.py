import numpy as np
import pandas as pd
import pyomo.environ as pyo
from symbolic_regression import SymbolicRegressionModel
import os
import sys
import sympy as sp

# from utils import get_gurobi

# Get the directory where this python script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the location of the Results directory
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results")

rng_samples_split = np.random.default_rng(42)

print()
print("--------Hyperparameters used--------")
with open("dataset_name.txt", "r") as file:
    dataset_name = file.read().strip()

with open("model_type.txt", "r") as file:
    model_type = file.read().strip()
print("Model type of the MINLP =", model_type)

with open("solver.txt", "r") as file:
    solver = file.read().strip()
print("Solver used to solve the MINLP =", solver)
# Save solver name before it gets overwritten with solver object
solver_name = solver

with open("max_time.txt", "r") as file:
    max_time = file.read().strip()
print("Maximum time allowed for solver to run =", max_time)

with open("number_of_samples.txt", "r") as file:
    number_of_samples = int(file.read().strip())
print("Number of samples =", number_of_samples)

with open("fitness_metric.txt", "r") as file:
    fitness_metric = file.read().strip()
print("Fitness metric to minimize =", fitness_metric)

with open("max_depth.txt", "r") as file:
    depth = int(file.read().strip())
print("Maximum depth of tree =", depth)

with open("warm_start.txt", "r") as file:
    warm_start = file.read().strip()
print("Warm-start =", warm_start)

# Check if custom variable bounds are provided via txt files
custom_bounds = False
if os.path.exists("v_lo.txt") and os.path.exists("v_up.txt"):
    custom_bounds = True


df = pd.read_csv(
    os.path.join(SCRIPT_DIR, "Datasets/Feynman_all_depths", dataset_name), sep="\t"
)
X_df = df.iloc[:, :-1]
y_df = df.iloc[
    :, [-1]
]  # if you use df.iloc[:, -1] it will return a Series instead of a DataFrame

# Downsampling
all_idx = np.arange(len(df))  # array of indices of all data points

if len(df) > number_of_samples:
    train_idx = rng_samples_split.choice(len(df), number_of_samples, replace=False)
    X_train_df = X_df.iloc[train_idx]
    y_train_df = y_df.iloc[train_idx]
else:
    # Use all data for training if dataset is small
    train_idx = all_idx
    X_train_df = X_df
    y_train_df = y_df

test_idx = np.setdiff1d(all_idx, train_idx)  # indices of the test points
X_test_df = X_df.iloc[test_idx]
y_test_df = y_df.iloc[test_idx]
X_test_np = X_test_df.to_numpy()
y_test_np = y_test_df.to_numpy().reshape(-1, 1)

# Save train and test data along with indices from the original dataset
# The index displayed is the datafrmame's 0-based index, not the row number in the original tsv file
# row_number = pandas_df_index + 2 (to account for header and 0-based index)
os.makedirs(os.path.join(RESULTS_DIR, "Train_data"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "Test_data"), exist_ok=True)

# Create subdirectories for train and test data based on hyperparameters (without max_nodes in folder name)
train_data_subdir_name = f"train_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_samples{number_of_samples}"
train_data_subdir_path = os.path.join(RESULTS_DIR, "Train_data", train_data_subdir_name)
os.makedirs(train_data_subdir_path, exist_ok=True)

test_data_subdir_name = f"test_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_samples{number_of_samples}"
test_data_subdir_path = os.path.join(RESULTS_DIR, "Test_data", test_data_subdir_name)
os.makedirs(test_data_subdir_path, exist_ok=True)

# Create SR_models folder (without max_nodes in folder name)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "SR_models"), exist_ok=True)
sr_models_folder_name = f"SR_models_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_samples{number_of_samples}"
sr_models_folder_path = os.path.join(RESULTS_DIR, "SR_models", sr_models_folder_name)
os.makedirs(sr_models_folder_path, exist_ok=True)

num_nodes_allowed = []
sse_with_nodes_constrained = []
previous_solution = None  # Previous iteration's solution for warm-start

for max_num_nodes in range(3, 7):

    # Create experiment name with max_nodes for individual files
    experiment_name = f"{dataset_name}_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_max_nodes_{max_num_nodes}_samples{number_of_samples}"
    if custom_bounds:
        experiment_name += "_bounds_x10"

    train_with_data = pd.concat([X_train_df, y_train_df], axis=1)
    train_with_data.to_csv(
        os.path.join(train_data_subdir_path, f"train_data_{experiment_name}.csv"),
        index=True,
    )

    test_with_data = pd.concat([X_test_df, y_test_df], axis=1)
    test_with_data.to_csv(
        os.path.join(test_data_subdir_path, f"test_data_{experiment_name}.csv"),
        index=True,
    )

    print()
    feature_names = X_df.columns.tolist()  # Preserves order
    print("Dataset =", dataset_name)
    print("Feature names:", feature_names)
    print("Target name:", y_df.columns[0])

    data = pd.concat([X_train_df, y_train_df], axis=1)
    input_cols = list(X_train_df.columns)
    output_col = y_train_df.columns[0]
    print(data)

    # Compute variable bounds from training data
    v_lo_default = min(-100, X_train_df.min().min() - 1, y_train_df.min().min() - 1)
    v_up_default = max(100, X_train_df.max().max() + 1, y_train_df.max().max() + 1)

    # Check if custom variable bounds are provided via txt files
    if os.path.exists("v_lo.txt") and os.path.exists("v_up.txt"):
        with open("v_lo.txt", "r") as file:
            v_lo = float(file.read().strip())
        with open("v_up.txt", "r") as file:
            v_up = float(file.read().strip())
        print(f"Using custom variable bounds: v_lo = {v_lo}, v_up = {v_up}")
    else:
        v_lo = v_lo_default
        v_up = v_up_default
        print(f"Variable bounds: v_lo = {v_lo}, v_up = {v_up}")

    def build_model():
        m = SymbolicRegressionModel(
            data=data,
            input_columns=input_cols,
            output_column=output_col,
            tree_depth=depth,
            operators=["sum", "diff", "mult", "div", "square", "sqrt", "exp", "log"],
            var_bounds=(v_lo, v_up),
            constant_bounds=(-100, 100),
            model_type=model_type,
        )
        m.add_objective(fitness_metric)

        return m

    if __name__ == "__main__":
        solver = solver_name
        mdl = build_model()
        # mdl.add_tree_size_constraint(3)
        mdl.constrain_max_tree_size(max_num_nodes)

        # Warm-start for BARON: Initialize with previous solution
        if warm_start == "yes" and solver == "baron" and previous_solution is not None:
            print(f"\nApplying warm-start from previous iteration.")
            warm_start_count = 0
            for var in mdl.component_data_objects(pyo.Var):
                if var.name in previous_solution:
                    var.value = previous_solution[var.name]
                    warm_start_count += 1
            print(f"Initialized {warm_start_count} variables from previous solution.\n")

        if solver == "scip":
            os.makedirs(os.path.join(RESULTS_DIR, "scip_model_MINLP"), exist_ok=True)
            os.makedirs(os.path.join(RESULTS_DIR, "scip_res_MINLP"), exist_ok=True)
            os.makedirs(os.path.join(RESULTS_DIR, "scip_lst_MINLP"), exist_ok=True)
            os.makedirs(
                os.path.join(RESULTS_DIR, "scip_results_stats_MINLP"), exist_ok=True
            )
            os.makedirs(
                os.path.join(RESULTS_DIR, "scip_solver_log_MINLP"), exist_ok=True
            )

            solver = pyo.SolverFactory("gams")

            # Capture SCIP solver output to a file
            import sys

            scip_logfile = os.path.join(
                RESULTS_DIR, "scip_solver_log_MINLP", f"scip_log_{experiment_name}.log"
            )

            with open(scip_logfile, "w") as log_f:
                # Redirect stdout to capture solver output
                old_stdout = sys.stdout
                sys.stdout = log_f

                results = solver.solve(
                    mdl,
                    solver="scip",
                    tee=True,  # This will write to our redirected stdout
                    symbolic_solver_labels=True,
                    keepfiles=True,
                    tmpdir=os.path.join(RESULTS_DIR, "scip_temp"),
                    add_options=[f"option reslim= {max_time};"],  # in seconds
                )

                # Restore stdout
                sys.stdout = old_stdout

            # Also print to console
            print(f"SCIP solver log saved to {scip_logfile}")

            # Copy GAMS/SCIP output files from temp directory to desired location
            import glob
            import shutil

            temp_dir = os.path.join(RESULTS_DIR, "scip_temp")
            if os.path.exists(temp_dir):
                # Copy GAMS model file
                gms_files = glob.glob(os.path.join(temp_dir, "*.gms"))
                if gms_files:
                    latest_gms = max(gms_files, key=os.path.getmtime)
                    dest_gms = os.path.join(
                        RESULTS_DIR, "scip_model_MINLP", f"model_{experiment_name}.gms"
                    )
                    shutil.copy2(latest_gms, dest_gms)
                    print(f"Copied GAMS model to {dest_gms}")

                # Copy listing file
                lst_files = glob.glob(os.path.join(temp_dir, "*.lst"))
                if lst_files:
                    latest_lst = max(lst_files, key=os.path.getmtime)
                    dest_lst = os.path.join(
                        RESULTS_DIR, "scip_lst_MINLP", f"lst_{experiment_name}.lst"
                    )
                    shutil.copy2(latest_lst, dest_lst)
                    print(f"Copied GAMS listing to {dest_lst}")

                # Copy solution file
                sol_files = glob.glob(os.path.join(temp_dir, "*.sol"))
                if sol_files:
                    latest_sol = max(sol_files, key=os.path.getmtime)
                    dest_sol = os.path.join(
                        RESULTS_DIR, "scip_res_MINLP", f"res_{experiment_name}.sol"
                    )
                    shutil.copy2(latest_sol, dest_sol)
                    print(f"Copied solution file to {dest_sol}")

                # Copy results.dat if it exists
                results_dat = os.path.join(temp_dir, "results.dat")
                if os.path.exists(results_dat):
                    dest_res = os.path.join(
                        RESULTS_DIR, "scip_res_MINLP", f"results_{experiment_name}.dat"
                    )
                    shutil.copy2(results_dat, dest_res)
                    print(f"Copied results.dat to {dest_res}")

                # Copy resultsstat.dat if it exists
                resultsstat_dat = os.path.join(temp_dir, "resultsstat.dat")
                if os.path.exists(resultsstat_dat):
                    dest_stats = os.path.join(
                        RESULTS_DIR,
                        "scip_results_stats_MINLP",
                        f"stats_{experiment_name}.dat",
                    )
                    shutil.copy2(resultsstat_dat, dest_stats)
                    print(f"Copied resultsstat.dat to {dest_stats}")

        elif solver == "gurobi":
            os.makedirs(os.path.join(RESULTS_DIR, "gurobi_model_MINLP"), exist_ok=True)
            os.makedirs(os.path.join(RESULTS_DIR, "gurobi_res_MINLP"), exist_ok=True)
            os.makedirs(os.path.join(RESULTS_DIR, "gurobi_lst_MINLP"), exist_ok=True)
            os.makedirs(
                os.path.join(RESULTS_DIR, "gurobi_results_stats_MINLP"), exist_ok=True
            )
            os.makedirs(
                os.path.join(RESULTS_DIR, "gurobi_solver_log_MINLP"), exist_ok=True
            )

            solver = pyo.SolverFactory("gams")

            # Capture GUROBI solver output to a file
            import sys

            gurobi_logfile = os.path.join(
                RESULTS_DIR,
                "gurobi_solver_log_MINLP",
                f"gurobi_log_{experiment_name}.log",
            )

            with open(gurobi_logfile, "w") as log_f:
                # Redirect stdout to capture solver output
                old_stdout = sys.stdout
                sys.stdout = log_f

                results = solver.solve(
                    mdl,
                    solver="gurobi",
                    tee=True,  # This will write to our redirected stdout
                    symbolic_solver_labels=True,
                    keepfiles=True,
                    tmpdir=os.path.join(RESULTS_DIR, "gurobi_temp"),
                    add_options=[f"option reslim= {max_time};"],  # in seconds
                )

                # Restore stdout
                sys.stdout = old_stdout

            # Also print to console
            print(f"GUROBI solver log saved to {gurobi_logfile}")

            # Copy GAMS/GUROBI output files from temp directory to desired location
            import glob
            import shutil

            temp_dir = os.path.join(RESULTS_DIR, "gurobi_temp")
            if os.path.exists(temp_dir):
                # Copy GAMS model file
                gms_files = glob.glob(os.path.join(temp_dir, "*.gms"))
                if gms_files:
                    latest_gms = max(gms_files, key=os.path.getmtime)
                    dest_gms = os.path.join(
                        RESULTS_DIR,
                        "gurobi_model_MINLP",
                        f"model_{experiment_name}.gms",
                    )
                    shutil.copy2(latest_gms, dest_gms)
                    print(f"Copied GAMS model to {dest_gms}")

                # Copy listing file
                lst_files = glob.glob(os.path.join(temp_dir, "*.lst"))
                if lst_files:
                    latest_lst = max(lst_files, key=os.path.getmtime)
                    dest_lst = os.path.join(
                        RESULTS_DIR, "gurobi_lst_MINLP", f"lst_{experiment_name}.lst"
                    )
                    shutil.copy2(latest_lst, dest_lst)
                    print(f"Copied GAMS listing to {dest_lst}")

                # Copy solution file
                sol_files = glob.glob(os.path.join(temp_dir, "*.sol"))
                if sol_files:
                    latest_sol = max(sol_files, key=os.path.getmtime)
                    dest_sol = os.path.join(
                        RESULTS_DIR, "gurobi_res_MINLP", f"res_{experiment_name}.sol"
                    )
                    shutil.copy2(latest_sol, dest_sol)
                    print(f"Copied solution file to {dest_sol}")

                # Copy results.dat if it exists
                results_dat = os.path.join(temp_dir, "results.dat")
                if os.path.exists(results_dat):
                    dest_res = os.path.join(
                        RESULTS_DIR,
                        "gurobi_res_MINLP",
                        f"results_{experiment_name}.dat",
                    )
                    shutil.copy2(results_dat, dest_res)
                    print(f"Copied results.dat to {dest_res}")

                # Copy resultsstat.dat if it exists
                resultsstat_dat = os.path.join(temp_dir, "resultsstat.dat")
                if os.path.exists(resultsstat_dat):
                    dest_stats = os.path.join(
                        RESULTS_DIR,
                        "gurobi_results_stats_MINLP",
                        f"stats_{experiment_name}.dat",
                    )
                    shutil.copy2(resultsstat_dat, dest_stats)
                    print(f"Copied resultsstat.dat to {dest_stats}")

        elif solver == "baron":
            os.makedirs(os.path.join(RESULTS_DIR, "baron_model_MINLP"), exist_ok=True)
            baron_model_subdir_name = f"model_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_samples{number_of_samples}"
            baron_model_subdir_path = os.path.join(
                RESULTS_DIR, "baron_model_MINLP", baron_model_subdir_name
            )
            os.makedirs(baron_model_subdir_path, exist_ok=True)

            mdl.write(
                os.path.join(baron_model_subdir_path, f"{experiment_name}.bar"),
                format="bar",
            )

            os.makedirs(os.path.join(RESULTS_DIR, "baron_res_MINLP"), exist_ok=True)
            os.makedirs(os.path.join(RESULTS_DIR, "baron_log_MINLP"), exist_ok=True)
            os.makedirs(
                os.path.join(RESULTS_DIR, "baron_summaries_MINLP"), exist_ok=True
            )
            os.makedirs(
                os.path.join(RESULTS_DIR, "baron_timefiles_MINLP"), exist_ok=True
            )

            # Create subdirectory for baron results based on hyperparameters
            baron_res_subdir_name = f"res_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_samples{number_of_samples}"
            baron_res_subdir_path = os.path.join(
                RESULTS_DIR, "baron_res_MINLP", baron_res_subdir_name
            )
            os.makedirs(baron_res_subdir_path, exist_ok=True)

            # Create subdirectory for baron logs based on hyperparameters
            baron_log_subdir_name = f"log_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_samples{number_of_samples}"
            baron_log_subdir_path = os.path.join(
                RESULTS_DIR, "baron_log_MINLP", baron_log_subdir_name
            )
            os.makedirs(baron_log_subdir_path, exist_ok=True)

            # Create subdirectory for baron summaries based on hyperparameters
            baron_summaries_subdir_name = f"summaries_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_samples{number_of_samples}"
            baron_summaries_subdir_path = os.path.join(
                RESULTS_DIR, "baron_summaries_MINLP", baron_summaries_subdir_name
            )
            os.makedirs(baron_summaries_subdir_path, exist_ok=True)
            # Create subdirectory for baron timefiles based on hyperparameters
            baron_timefiles_subdir_name = f"timefiles_{model_type}_{fitness_metric}_{solver_name}_max_time_{max_time}_maxdepth{depth}_samples{number_of_samples}"
            baron_timefiles_subdir_path = os.path.join(
                RESULTS_DIR, "baron_timefiles_MINLP", baron_timefiles_subdir_name
            )
            os.makedirs(baron_timefiles_subdir_path, exist_ok=True)
            solver = pyo.SolverFactory("baron")
            solver.options["MaxTime"] = max_time
            solver.options["CplexLibName"] = r"C:\GAMS\48\cplex2211.dll"
            # pyomo deactivates the creation of summary file, even though baron creates it by default
            solver.options["summary"] = 1
            solver.options["SumName"] = os.path.join(
                baron_summaries_subdir_path, f"summary_{experiment_name}"
            )

            results = solver.solve(
                mdl,
                tee=True,
                symbolic_solver_labels=True,
                keepfiles=True,
                solnfile=os.path.join(baron_res_subdir_path, f"res_{experiment_name}"),
                logfile=os.path.join(
                    baron_log_subdir_path, f"log_{experiment_name}.log"
                ),
            )

            # Copy the tim file from temp directory to desired location
            import glob
            import shutil

            temp_dir = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
            tim_pattern = os.path.join(temp_dir, "tmp*.baron.tim")
            tim_files = glob.glob(tim_pattern)
            if tim_files:
                # Get the most recently modified tim file
                latest_tim = max(tim_files, key=os.path.getmtime)
                dest_tim = os.path.join(
                    baron_timefiles_subdir_path, f"tim_{experiment_name}"
                )
                shutil.copy2(latest_tim, dest_tim)
                print(f"Copied tim file to {dest_tim}")

            # Save solution for warm-start in next iteration
            if warm_start == "yes":
                previous_solution = {}
                for var in mdl.component_data_objects(pyo.Var):
                    if var.value is not None:
                        previous_solution[var.name] = var.value
                print(
                    f"\nSaved {len(previous_solution)} variable values for warm-start in next iteration."
                )

        mdl.constant_val.pprint()
        print(mdl.get_parity_plot_data())
        print("Assigned operators to nodes: ", mdl.get_selected_operators())

        # Get the objective value from the solved model
        objective_value = pyo.value(mdl.sse)
        print(f"Objective value: {objective_value}")

        parity_data = mdl.get_parity_plot_data()

        # Append predictions to train data CSV
        train_data_path = os.path.join(
            train_data_subdir_path, f"train_data_{experiment_name}.csv"
        )
        train_df = pd.read_csv(train_data_path, index_col=0)
        train_df["prediction"] = parity_data["prediction"].values
        train_df["square_of_error"] = parity_data["square_of_error"].values
        train_df.to_csv(train_data_path, index=True)

        # Calculate NRMSE for training set
        train_actuals = parity_data["sim_data"].values
        train_predictions = parity_data["prediction"].values
        sse_minlp = np.sum((train_actuals - train_predictions) ** 2)
        rmse_minlp = np.sqrt(np.mean((train_actuals - train_predictions) ** 2))
        y_range = train_actuals.max() - train_actuals.min()
        nrmse_minlp = rmse_minlp / y_range if y_range != 0 else 0.0

        # SSE calculated using the true derived model
        parity_data_sim = mdl.get_parity_plot_data()["sim_data"].values
        sse_true = sum(
            [
                (
                    mdl.samples[sample_idx].compare_node_values().loc[1, "True Value"]
                    - parity_data_sim[idx]
                )
                ** 2
                for idx, sample_idx in enumerate(mdl.samples)
            ]
        )

        # Calculate SSE for every node
        SSE_node_list = []
        # Get the first sample index to iterate over the nodes
        first_sample_idx = next(iter(mdl.samples))
        for node_id in mdl.samples[first_sample_idx].compare_node_values().index:
            SSE_node = float(
                sum(
                    [
                        (
                            mdl.samples[sample_idx]
                            .compare_node_values()
                            .loc[node_id, "Computed Value"]
                            - mdl.samples[sample_idx]
                            .compare_node_values()
                            .loc[node_id, "True Value"]
                        )
                        ** 2
                        for sample_idx in mdl.samples
                    ]
                )
            )
            SSE_node_list.append(SSE_node)
            print(f"Node {node_id}: SSE = {SSE_node}")

        print("\nSSE_node_list =", SSE_node_list)

        expr = mdl.selected_tree_to_expression()
        expr = expr.sympy_expression
        print("SR model:", expr)

        selected_operators = mdl.get_selected_operators()

        # Extract constant values from the model
        constant_values = [pyo.value(mdl.constant_val[i]) for i in mdl.constant_val]
        print("Constant values:", constant_values)

        # Substitute constant values into the expression
        constant_substitutions = {}
        for i in mdl.constant_val:
            constant_substitutions[sp.Symbol(f"c_{i}")] = pyo.value(mdl.constant_val[i])

        SR_model = expr.subs(constant_substitutions)
        print("SR model with constant values substituted:", SR_model)

        # Get constants that actually appear in the expression
        expr_symbols = expr.free_symbols
        constants_in_expr = []
        for i in mdl.constant_val:
            c_symbol = sp.Symbol(f"c_{i}")
            if c_symbol in expr_symbols:
                constants_in_expr.append((i, pyo.value(mdl.constant_val[i])))

        # Format constant values as a list string [c1: value, c2: value, ...] (only constants in expression)
        constant_values_str = (
            "[" + ", ".join([f"c{i}: {val}" for i, val in constants_in_expr]) + "]"
        )

        # Make predictions on the test data first
        feature_vars = [
            sp.Symbol(f"x{i}") for i in range(1, len(X_test_df.columns) + 1)
        ]
        SR_model_func = sp.lambdify(feature_vars, SR_model, modules="numpy")

        test_predictions = np.array(
            [SR_model_func(*row) for row in X_test_df.to_numpy()]
        )
        test_actuals = y_test_df.values.flatten()
        sse_test = np.sum((test_actuals - test_predictions) ** 2)

        # Check if SR model's prediction is correct
        sr_prediction_correct = 1 if abs(sse_minlp - sse_true) < 1e-6 else 0

        # Check if SR model generalizes on the test set
        sr_model_good_on_test = 1 if abs(sse_test) < 1e-1 else 0

        # Check if all node values are correct
        node_values_correct = 1 if all(abs(sse) < 1e-6 for sse in SSE_node_list) else 0

        law_probably_found = (
            1
            if (
                sr_prediction_correct == 1
                and sr_model_good_on_test == 1
                and node_values_correct == 1
            )
            else 0
        )

        # Create individual CSV file for this experiment
        individual_csv_path = os.path.join(
            sr_models_folder_path, f"SR_model_{experiment_name}.csv"
        )

        # Create new row with experiment data
        new_row = pd.DataFrame(
            {
                "experiment_name": [experiment_name],
                "max_nodes": [max_num_nodes],
                "expression": [
                    f'="{str(expr)}"'
                ],  # Excel formula format to display as text
                "SR_model": [
                    f'="{str(SR_model)}"'
                ],  # Excel formula format to display as text
                "constant_values": [constant_values_str],
                "nodes_assignments": [str(selected_operators)],
                "var_bounds": [f"({v_lo}, {v_up})"],
                "nrmse_minlp": [nrmse_minlp],
                "sse_minlp": [sse_minlp],
                "objective": [objective_value],
                "sse_true": [sse_true],
                "sr_prediction_correct": [sr_prediction_correct],
                "SSE_node_list": [str(SSE_node_list)],
                "node_values_correct": [node_values_correct],
                "sse_test": [sse_test],
                "sr_model_good_on_test": [sr_model_good_on_test],
                "law_probably_found": [law_probably_found],
            }
        )

        # Save individual experiment to its own CSV file
        new_row.to_csv(individual_csv_path, index=False, quoting=1)

        # Append predictions to test data CSV
        test_data_path = os.path.join(
            test_data_subdir_path, f"test_data_{experiment_name}.csv"
        )
        test_df = pd.read_csv(test_data_path, index_col=0)
        test_df["prediction"] = test_predictions
        test_df["square_of_error"] = (test_actuals - test_predictions) ** 2
        test_df.to_csv(test_data_path, index=True)

        num_nodes_allowed.append(max_num_nodes)
        sse_with_nodes_constrained.append(sse_true)
        print(
            f"Sum of squares of error for a maximum of {max_num_nodes} nodes is {sse_with_nodes_constrained[-1]}"
        )

        if np.isclose(sse_with_nodes_constrained[-1], 0, rtol=1e-6):
            break
