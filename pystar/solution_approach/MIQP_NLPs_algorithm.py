import logging
import time

import numpy as np
import pandas as pd
from typing import Callable, Optional
import pyomo.environ as pyo #used for configuring baron
from pyomo.contrib import appsi #Auto-Persistent Pyomo Solver Interfaces

from pystar import setup_logger
from pystar.core.expression_tree import ExpressionTree

LOGGER = logging.getLogger(__name__)
setup_logger(log_level="info")

# Default termination tolerances of BARON (EpsA= 1e-6, EpsR= 1e-6): BARON terminates if |U - L| <= EpsA or |U - L| <= EpsR max(|L|, |U|)
# (SCIP's defaults are limits/absgap = limits/gap = 0, i.e. U = L up to numerics/epsilon = 1e-9)
EPS_A = 1e-6
EPS_R = 1e-6


def gap_closed(lb, UB):
    """True if lb >= UB within the termination tolerances, i.e. a tree with lower bound lb cannot improve UB by more
    than the tolerances. The tolerances apply only to finite bounds"""
    return lb >= UB or (-float("inf") < lb < UB < float("inf") and (UB - lb <= EPS_A or UB - lb <= EPS_R * max(abs(lb), abs(UB))))

############ Main function ############
# time_limit is the total time (s) of the MIQP and the NLPs: the MIQP is capped at time_limit and each NLP gets the time left
# solve_nlp(model, solution_index, time_limit) solves the NLP of a tree of the pool. By default, SCIP (solve_with_scip)
# If no_enumeration=False, then all k NLP's will be solved
# If no_zero_cst=True, then the algorithm doesn't terminate if the SR expression has a constant that equals zero
# gurobi_options are added to (or overwrite) the Gurobi options of the MIQP, e.g. {"IntFeasTol": 1e-6}
# solve_miqp(solver, model) solves the MIQP with the configured Gurobi solver, e.g. to keep its files. By default, solver.solve(model)
def MIQP_NLPs_algorithm(m, k, time_limit, solve_nlp: Callable | None = None, UB: float | None = None, no_enumeration: bool = True, no_zero_cst: bool = False, gurobi_options: dict | None = None, solve_miqp: Callable | None = None):

    start = time.perf_counter()
    if solve_nlp is None:  # Default NLP solver: SCIP with the time left
        solve_nlp = lambda model, solution_index, time_left: solve_with_scip(model, time_left)
    UB = float("inf") if UB is None else UB # incumbent, possibly from a previous run of algorithm

    pool_of_solutions, miqp_optimal, miqp_info = get_MILP_solution_pool(m, k, time_limit, gurobi_options, solve_miqp)
    all_nlps_solved_globally = True

    shrank_pool = []

    # One clone for all NLPs to make code faster
    nlp_model = m.clone()
    var_list = list(nlp_model.component_data_objects(pyo.Var, descend_into=True))
    initial_values = [v.value for v in var_list]
    best_values, best_tree = None, None
    optimized_trees = []  # Optimized trees from solved NLPs
    
    # Expression trees of the whole MIQP pool
    expression_trees = [
        str(ExpressionTree([m.find_component(name).index() for name in solution["variables"]]))
        for solution in pool_of_solutions
    ]

    for i in range(len(pool_of_solutions)):
        #For the first solution (i=0), lb_0 is a valid LB of MINLP

        # Each NLP gets the time left of time_limit. Without time left, the rest of the trees of the pool are not solved
        time_left = time_limit - (time.perf_counter() - start)
        if time_left <= 0:
            LOGGER.warning("\n Time limit reached after solving %s NLPs", i)
            break

        nlp_objective, et, nlp_solved_globally = solve_NLP(nlp_model, pool_of_solutions, i, solve_nlp, time_left, var_list, initial_values)
        all_nlps_solved_globally = all_nlps_solved_globally and nlp_solved_globally

        # Tree with the values of its constants. The values of the best NLP so far are saved (the first one if tied)
        optimized_trees.append(str(nlp_model.selected_tree_to_expression()) if nlp_objective < float("inf") else None)
        if nlp_objective < min(shrank_pool, default=float("inf")):
            best_values = [v.value for v in var_list]
            best_tree = nlp_model.selected_tree_to_expression()

        UB = min(UB, nlp_objective) # get the UB for iteration i

        LOGGER.debug("\n The UB so far: %s", UB)
        shrank_pool.append(nlp_objective)

        if no_enumeration and i + 1 < len(pool_of_solutions):
            if gap_closed(pool_of_solutions[i+1]["objective"], UB): #check if lb_i+1 >= min(UB, UB_i) within the tolerances
                LOGGER.debug("\n For %s-th iteration with objective/lb_i+1 from pool: %s and NLP objective: %s", i, pool_of_solutions[i+1]["objective"], nlp_objective)
                
                # Values of the constants of the best tree so far
                has_zero_constant = best_tree is not None and any(
                    np.isclose(
                        value,
                        0.0,
                        atol=1e-4,
                        rtol=0.0,
                    )
                    for value in best_tree.cst_values.values()
                )           

                if has_zero_constant and no_zero_cst:
                    LOGGER.debug("\n Not breaking because the current best model has a zero constant.")
                else:                    
                    LOGGER.info("\n Breaking the loop as the %s-th objective from pool is greater than or equal to the current UB.", i+2)
                    LOGGER.info("\n Solved %s NLPs", i+2) 
                    break

        LOGGER.debug("\n shrank pool so far: %s", shrank_pool)

    LOGGER.debug("\n shrank pool of NLP objectives: %s", shrank_pool)


    # No NLP is solved if the MIQP used all the time
    optimal_objective = min(shrank_pool, default=float("inf"))
    optimal_solution_index = shrank_pool.index(optimal_objective) if shrank_pool else None

    # Load the best NLP (its tree and its values) into nlp_model, e.g. for the checks of the best model.
    # best_model is None if no NLP has a solution
    best_model = None
    if best_values is not None:
        fix_tree(nlp_model, pool_of_solutions[optimal_solution_index])
        for v, value in zip(var_list, best_values):
            v.set_value(value, skip_validation=True)
        best_model = nlp_model

    lb_k = pool_of_solutions[-1]["objective"]   # Gurobi sorts the pool best to worst
    # The trees of the pool that were not solved (pruned or out of time) need lb >= UB (within the tolerances). As the
    # pool is sorted, the first of them is checked
    lb_unsolved = pool_of_solutions[len(shrank_pool)]["objective"] if len(shrank_pool) < len(pool_of_solutions) else float("inf")
    proof_of_global_optimality = miqp_optimal and all_nlps_solved_globally and gap_closed(lb_unsolved, UB) and (gap_closed(lb_k, UB) or len(pool_of_solutions) < k)
    if not proof_of_global_optimality:
        LOGGER.warning("\n Global optimality is not certified. UB = %s, lb_k = %s", UB, lb_k)

    if proof_of_global_optimality:
        # UB, or the lb of the trees not solved (in the pool, or outside a full pool) if it is below UB within the tolerances
        LB = min(UB, lb_unsolved, lb_k if len(pool_of_solutions) == k else float("inf"))
    elif miqp_optimal:
        LB = pool_of_solutions[0]["objective"]
    else:
        LB = miqp_info["MIQP_best_bound"]

    if proof_of_global_optimality:
        termination_condition = "optimal" if UB < float("inf") else "infeasible"
    elif time.perf_counter() - start >= time_limit:
        termination_condition = "maxTimeLimit"
    else:
        termination_condition = "notCertified"  # e.g. lb_k < UB (k too small) or an NLP not solved to global optimality

    info = {
        "LB": LB,
        "termination_condition": termination_condition,
        "pool_size": len(pool_of_solutions),
        "pool_objectives": [solution["objective"] for solution in pool_of_solutions],
        "optimized_trees": optimized_trees,
        **miqp_info,
    }

    return optimal_objective, optimal_solution_index, best_model, expression_trees, shrank_pool, proof_of_global_optimality, info

# Step 1
def get_MILP_solution_pool(m, k, MIQP_time_limit, gurobi_options: dict | None = None, solve_miqp: Callable | None = None):
    m1 = m.clone()
    m1.relax_nonconvex_constraints()

    # Solve the MIQP with gurobi
    print("\n Clone of disjunctive program created and nonconvexities in each disjunct were relaxed. Using hull formulation, the MIQP is now built. \n")
    solver = gurobi_solver(m1, k, MIQP_time_limit, gurobi_options)
    results = solver.solve(m1) if solve_miqp is None else solve_miqp(solver, m1)

    # Check if MIQP was solved to global optimality
    miqp_optimal = (results.termination_condition == appsi.base.TerminationCondition.optimal) # True or False
    if not miqp_optimal:
        LOGGER.warning("\n MIQP ended with %s: the pool may not hold the k best solutions", results.termination_condition)

    # MIQP statistics. The best bound equals lb_0 if the MIQP is solved to optimality (MIPGap = 0)
    miqp_info = {
        "MIQP_solver": "gurobi_v" + ".".join(map(str, solver.version())),  # e.g. gurobi_v13.0.2
        "MIQP_termination_condition": results.termination_condition.name,
        "MIQP_best_bound": results.best_objective_bound,
        "MIQP_bnb_nodes": solver.get_model_attr("NodeCount"),
        "MIQP_time": solver.get_model_attr("Runtime"),
    }

    # Get MIQP pool and check if it is actually created
    pool_of_solutions = get_pool_of_solutions_from_solver_results(solver, results)
    if not pool_of_solutions:
        raise RuntimeError("MIQP returned no solutions")


    return pool_of_solutions, miqp_optimal, miqp_info

# Step 2
def fix_tree(m2, solution):
    """Fixes select_operator to the tree of a solution of the pool: 1 for its operators, 0 for the rest"""
    m2.select_operator.fix(0)
    for name in solution["variables"]:
        m2.find_component(name).fix(1)


# m2 is the NLP model of all trees of the pool, var_list its variables and initial_values their values before any NLP
def solve_NLP(m2, pool_of_solutions, solution_index, solve_nlp, time_limit, var_list, initial_values):
    # Same starting point as a clone of the SR model: the values of the previous NLP are discarded
    LOGGER.debug("\n Resetting the variables to their initial values.")
    for v, value in zip(var_list, initial_values):
        v.set_value(value, skip_validation=True)

    LOGGER.debug("\n Fixing selected operators to 1 (and the rest to 0) for the %s-th solution", solution_index)
    fix_tree(m2, pool_of_solutions[solution_index])
    #m2.select_operator.pprint()

    et = ExpressionTree(m2.get_selected_operators())
    LOGGER.info("\n Expression tree structure for the %s-th solution: %s", solution_index, et)
    #m2.select_operator.pprint()

    LOGGER.debug("\n Solving NLP for the %s-th solution", solution_index)
    results = solve_nlp(m2, solution_index, time_limit)

    nlp_objective = float("inf")
    nlp_solved_globally = True
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        nlp_objective = pyo.value(m2.penalty_metrics["sse"])
    elif results.solver.termination_condition in (pyo.TerminationCondition.infeasible, pyo.TerminationCondition.infeasibleOrUnbounded):
        nlp_objective = float("inf")
    else:
        # e.g. if solver hits max time limit: its best solution is still a valid UB
        # Pyomo sets results.problem.upper_bound only if SCIP is optimal, so a best solution exists (and is loaded
        # in m2) if SCIP's primal bound is < 1e20 (SCIP's infinity)
        if results.problem.upper_bound < float("inf") or getattr(results.solver, "primal_bound", float("inf")) < 1e20:
            nlp_objective = pyo.value(m2.penalty_metrics["sse"])
        nlp_solved_globally = False
        LOGGER.warning("\n NLP %s ended with %s, bounds [%s, %s]", solution_index,
                       results.solver.termination_condition,
                       getattr(results.solver, "dual_bound", results.problem.lower_bound), nlp_objective)

    return nlp_objective, et, nlp_solved_globally #, results


def solve_with_scip(m, MaxTime: Optional[float] = 100):
    solver = pyo.SolverFactory("scip")
    solver.options["limits/time"] = MaxTime

    LOGGER.debug("\n SCIP solver configured.")

    results = solver.solve(
        m,
        #symbolic_solver_labels=True,
        #keepfiles=False,        
    )

    return results


def solve_with_baron(m, MaxTime: Optional[float] = 100):
    solver = pyo.SolverFactory("baron")
    solver.options["MaxTime"] = MaxTime
    solver.options["CplexLibName"] = r"C:\baron\cplex2212.dll"
    LOGGER.debug("\n Baron solver configured.")

    results = solver.solve(
        m,
        #tee=True, # If True, print log output
        symbolic_solver_labels=True,
        keepfiles=False, # True to keep solver's files
        #solnfile=f"res.sol",
        #logfile=f"log.log",
    )

    return results

############# Sub-routines used in step 1 (MIQP) ##################
def gurobi_solver(m, k, MIQP_time_limit, gurobi_options: dict | None = None):
    #Solver and options 
    solver = appsi.solvers.Gurobi()  # Create an Appsi Gurobi solver instance
    solver.config.stream_solver = False  # If true, print/stream solver's log output while solving
    solver.gurobi_options["LogToConsole"]=0
    solver.gurobi_options["OutputFlag"]=0
    
    # Useful when you want to load in Pyomo model a solution from the pool:
    solver.config.load_solution = False  # Don't load solution in Pyomo model m
    solver.gurobi_options["TimeLimit"] = MIQP_time_limit
    solver.gurobi_options["MIPGap"] = 0  # needed for lb_k to bound every tree outside the pool
    solver.gurobi_options["PoolSolutions"] = k  # Store up to three feasible solutions in the pool
    #Search modes
    # 0: don't search for more after finding the optimal
    # 1: search for more after finding the optimal
    # 2: Search for the k-best solutions
    solver.gurobi_options["PoolSearchMode"] = 2  
    # solver.gurobi_options["PoolGapAbs"] = 1e-9  # Max allowed absolute gap between the best objective value and the rest obj values in the pool
    # solver.gurobi_options["PoolGap"] = 0  # Max allowed relative gap between the best objective value and the rest obj values in the pool
    # Extra options, e.g. {"IntFeasTol": 1e-6, "OutputFlag": 1, "LogFile": "miqp.log"} to keep the log in a file
    solver.gurobi_options.update(gurobi_options or {})
    LOGGER.debug("\n Gurobi solver configured.")
    return solver


def get_pool_of_solutions_from_solver_results(solver, results):
    pool_of_solutions = []
    sol_count = solver.get_model_attr("SolCount")  # Get the number of solutions in the pool
    # SolutionNumber is the index of the solution under inspection. Solutions are stored from best to worse
    # get_gurobi_param_info() returns a 6-element tuple containing information about the Gurobi parameter
    #initial_solution_number = solver.get_gurobi_param_info("SolutionNumber")[2]  # Index 2 depicts the current value of SolutionNumber

    for i in range(sol_count):
        # Variable values for solution i
        primal_vars = results.solution_loader.get_primals(solution_number=i)

        solver.set_gurobi_param("SolutionNumber", i) # Set the current solution number to i
        # Get the objective value for the current solution from the pool
        objective = solver.get_model_attr("PoolNObjVal")
       
        pool_of_solutions.append({
            "objective": objective,
            "variables": {
                var.name: value
                for var, value in primal_vars.items() if "select_operator" in var.name and value > 1e-3
            },
        })
        
    #solver.set_gurobi_param("SolutionNumber", initial_solution_number)

    pool_of_solutions_df = pd.DataFrame({
    solution_number: solution["variables"]
    for solution_number, solution in enumerate(pool_of_solutions)
    }).fillna(0)

    LOGGER.debug("\n Pool of solutions created: %s", pool_of_solutions)
    LOGGER.debug("\n Pool of solutions DataFrame: %s", pool_of_solutions_df)
    return pool_of_solutions