"""Reiterates the MIQP-NLPs algorithm until the best tree is certified as globally optimal or the time runs out.
Each round gets the UB of the previous rounds and its MIQP excludes the trees already solved with no-good cuts,
so its pool holds the next k best trees
"""

import logging
import time
from typing import Callable

import pyomo.environ as pyo
from pyomo.contrib import appsi

from pystar import setup_logger
from pystar.solution_approach.MIQP_NLPs_algorithm import MIQP_NLPs_algorithm, gap_closed, solve_with_scip

LOGGER = logging.getLogger(__name__)
setup_logger(log_level="info")

# NLP solved to global optimality (or proven infeasible), as in solve_NLP of MIQP_NLPs_algorithm
NLP_SOLVED_GLOBALLY = (
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.infeasible,
    pyo.TerminationCondition.infeasibleOrUnbounded,
)
MIQP_INFEASIBLE = (appsi.base.TerminationCondition.infeasible, appsi.base.TerminationCondition.infeasibleOrUnbounded)


############ Main function ############
# Same arguments and outputs as MIQP_NLPs_algorithm. kwargs are passed to it, e.g. no_enumeration or gurobi_options
# time_limit is the total time of all rounds: each round gets the time left
# The rounds stop if a round is certified or runs out of time. A tree whose NLP is not solved globally (e.g. SCIP
# stops for a reason other than optimality or infeasibility) is excluded from the next rounds with its lower bound,
# max(lb of the pool, dual bound of SCIP): the LB is at most this bound and, if this bound is < UB (beyond the
# tolerances of gap_closed), the best tree is not certified
# The NLPs are numbered over all rounds, in solve_nlp(model, solution_index, time_left) (e.g. for its files) and
# in the outputs. info["rounds"] is the number of MIQPs solved
def MIQP_NLPs_reiterated(m, k, time_limit, solve_nlp: Callable | None = None, solve_miqp: Callable | None = None, **kwargs):

    start = time.perf_counter()
    if solve_nlp is None:  # Default NLP solver: SCIP with the time left
        solve_nlp = lambda model, solution_index, time_left: solve_with_scip(model, time_left)

    solved_trees = []  # Selected operators of the tree of each solved NLP of all rounds
    not_solved_globally = {}  # Index of each NLP not solved globally: dual bound of SCIP (-inf if unknown)
    miqps = []  # Termination condition, time and B&B nodes of the MIQP of each round

    def solve_nlp_all_rounds(model, solution_index, time_left):
        solved_trees.append(set(model.get_selected_operators()))  # The tree of the NLP is fixed in model
        results = solve_nlp(model, len(solved_trees) - 1, time_left)
        if results.solver.termination_condition not in NLP_SOLVED_GLOBALLY:
            dual_bound = getattr(results.solver, "dual_bound", None)
            not_solved_globally[len(solved_trees) - 1] = -float("inf") if dual_bound is None else dual_bound
        return results

    # One no-good cut per solved tree: in next iteration of algorithm 
    # at least one select_operator of the solved tree is not selected so the next pool holds the next k best trees.
    # Only the solved tree is excluded: each active node has one operator and its children are fixed by it
    # (active_node_active_operator, left/right_child_presence), so no feasible tree strictly contains a solved tree.
    # m1 is the clone of m for the MIQP, so the NLPs don't have the cuts
    def solve_miqp_without_solved_trees(solver, m1):
        m1.no_good_cuts = pyo.ConstraintList()
        for tree in solved_trees:
            m1.no_good_cuts.add(sum(m1.select_operator[i] for i in tree) <= len(tree) - 1)
        results = solver.solve(m1) if solve_miqp is None else solve_miqp(solver, m1)
        miqps.append((results.termination_condition, solver.get_model_attr("Runtime"), solver.get_model_attr("NodeCount")))
        return results

    UB, LB = float("inf"), -float("inf")
    excluded_LB = float("inf")  # Lower bound of the excluded trees whose NLPs are not solved globally
    best_index, best_model, info = None, None, {}
    trees, pool_objectives, nlp_objectives, optimized_trees = [], [], [], []  # Solved trees of all rounds
    unsolved_trees, unsolved_objectives = [], []  # Trees of the pool of the last round that were not solved
    while True:
        offset = len(solved_trees)
        try:
            objective, index, model, pool_trees, shrank_pool, certified, info = MIQP_NLPs_algorithm(
                m,
                k,
                time_limit - (time.perf_counter() - start),
                solve_nlp=solve_nlp_all_rounds,
                UB=UB,
                solve_miqp=solve_miqp_without_solved_trees,
                **kwargs,
            )
        except RuntimeError as error:
            # MIQP of a later round without solutions: no tree is left (certified) or out of time
            if not info or str(error) != "MIQP returned no solutions":
                raise
            certified = miqps[-1][0] in MIQP_INFEASIBLE
            if certified:
                LB = max(LB, min(UB, excluded_LB))
                termination_condition = "optimal" if UB < float("inf") else "infeasible"
            else:
                termination_condition = "maxTimeLimit" if time.perf_counter() - start >= time_limit else "notCertified"
            LOGGER.info("\n Round %s: the MIQP has no solutions (%s)", len(miqps), miqps[-1][0].name)
            break

        # Trees of the pool are solved in order, so the solved ones are the first len(shrank_pool)
        solved = len(shrank_pool)
        if objective < UB:  # The best tree so far is of this round
            best_index, best_model = offset + index, model
        UB = min(UB, objective)
        # info["LB"] bounds the trees of this MIQP. The trees of the previous rounds (excluded) are >= UB if their NLPs
        # are solved globally and >= excluded_LB otherwise
        if info["LB"] is not None:
            LB = max(LB, min(info["LB"], UB, excluded_LB))
        for i in range(offset, len(solved_trees)):
            if i in not_solved_globally:
                excluded_LB = min(excluded_LB, max(info["pool_objectives"][i - offset], not_solved_globally[i]))
        trees += pool_trees[:solved]
        pool_objectives += info["pool_objectives"][:solved]
        unsolved_trees, unsolved_objectives = pool_trees[solved:], info["pool_objectives"][solved:]
        nlp_objectives += shrank_pool
        optimized_trees += info["optimized_trees"]
        termination_condition = info["termination_condition"]
        LOGGER.info("\n Round %s: %s, UB = %s, LB = %s, %s NLPs solved", len(miqps), termination_condition, UB, LB, len(solved_trees))

        if termination_condition != "notCertified":
            break
        if time.perf_counter() - start >= time_limit:
            termination_condition = "maxTimeLimit"
            break

    # A round certifies only its own NLPs: the excluded trees whose NLPs are not solved globally have only a lower bound,
    # so they could be better than the best tree if their bound is < UB (beyond the tolerances)
    if not gap_closed(excluded_LB, UB):
        certified = False
        if termination_condition in ("optimal", "infeasible"):
            termination_condition = "notCertified"

    info = {
        **info,  # MIQP solver and best bound of the last round with solutions
        "LB": LB,
        "termination_condition": termination_condition,
        "pool_size": len(trees) + len(unsolved_trees),
        "pool_objectives": pool_objectives + unsolved_objectives,
        "optimized_trees": optimized_trees,
        "MIQP_termination_condition": miqps[-1][0].name,
        "MIQP_bnb_nodes": sum(nodes for _, _, nodes in miqps),
        "MIQP_time": sum(runtime for _, runtime, _ in miqps),
        "rounds": len(miqps),
    }

    return UB, best_index, best_model, trees + unsolved_trees, nlp_objectives, certified, info
