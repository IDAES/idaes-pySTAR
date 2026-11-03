import logging
import time
from dataclasses import dataclass

import pandas as pd
import pyomo.environ as pyo  # Used for configuring baron
from pyomo.contrib.solver.solvers.gurobi.gurobi_persistent import GurobiPersistent
from pyomo.contrib.solver.solvers.gurobi.gurobi_direct_base import (
    NoFeasibleSolutionError,
)

from pystar import ExpressionTree, SymbolicRegressionModel
from pystar.core.utils import get_gurobi

LOGGER = logging.getLogger(__name__)

# Default termination tolerances of BARON (EpsA= 1e-6, EpsR= 1e-6)
# BARON terminates if |U - L| <= EpsA or |U - L| <= EpsR max(|L|, |U|)
# (SCIP's defaults are limits/absgap = limits/gap = 0,
# i.e. U = L up to numerics/epsilon = 1e-9)
EPS_A = 1e-6
EPS_R = 1e-6


def _solution_log(
    expression_num: int,
    lb: float,
    current_ub: float,
    solution_status: str,
    best_ub: float,
    comp_time: float,
):
    """Function to print the solver log

    Parameters
    ----------
    solution_num : int
        Solution number that is being evaluated

    lb : float
        Objective function value of the relaxed symbolic regression model
        corresponding to the solution that is being evaluated

    current_ub : float
        Objective function value of the solution of the NLP.

    solution_status : str
        Status of the NLP solve: OPT, TL, INF

    best_ub : float
        Best-known upper bound

    comp_time : float
        Total time elapsed.
    """

    # pylint: disable = logging-fstring-interpolation
    if expression_num == 0:
        LOGGER.info(
            f"{"Expr. Num."}    {"Lower bound"}    "
            f"{"Incumbent Sol./Status"}      {"Upper bound"}    {"  Time"}"
        )
        LOGGER.info(
            f"{"=" * 10}    {"=" * 11}    {"=" * 21}      {"=" * 11}    {"=" * 7}"
        )

    elif expression_num == -1:
        # This status implies that the model is solved to optimality
        LOGGER.info(f"{"=" * 77}")
        LOGGER.info(f"Optimal solution found (epsa: {EPS_A}, epsr: {EPS_R})")
        LOGGER.info(f"Best incumbent solution: {best_ub:11.5f}")
        LOGGER.info(f"Best bound             : {lb:11.5f}")
        LOGGER.info(f"Computational time     : {round(comp_time):7d}")

    elif expression_num == -3:
        # This status implies that the model has timed out
        LOGGER.info(f"{"=" * 77}")
        LOGGER.warning("Time limit reached.")
        LOGGER.info(f"Best incumbent solution: {best_ub:11.5f}")
        LOGGER.info(f"Best bound             : {lb:11.5f}")
        LOGGER.info(f"Computational time     : {round(comp_time):7d}")

    else:
        LOGGER.info(
            f"{expression_num:10d}    {lb:11.5f}    {current_ub:11.5f}"
            f"/{solution_status:>11}    {best_ub:11.5f}    {round(comp_time):7d}"
        )


@dataclass
class PoolAndPrune:
    """_summary_

    Returns
    -------
    _type_
        _description_
    """

    model: SymbolicRegressionModel
    pool_size: int
    overall_time_limit: float
    time_limit_nlp_solves: float = 100
    use_enumeration: bool = True
    use_zero_cst: bool = False
    relaxation_solver_options: dict | None = None
    nlp_solver_options: dict | None = None

    def __post_init__(self):
        self.best_incumbent = float("inf")
        self.best_bound = -float("inf")
        self._processed_expression_pool = {}
        self.no_good_cuts = []

        if self.relaxation_solver_options is None:
            self.relaxation_solver_options = {}

        if self.nlp_solver_options is None:
            self.nlp_solver_options = {}

        # Create a copy of the model, relax it, and store the relaxed version
        m1 = self.model.clone()
        m1.relax_nonconvex_constraints()
        m1.no_good_cuts = pyo.ConstraintList()
        LOGGER.info(
            "Created a clone of the MINLP; nonconvex constraints in each "
            "disjunct, if present, are relaxed."
        )
        self._relaxed_miqp_model = GurobiPersistent()
        self._relaxed_miqp_model.config.raise_exception_on_nonoptimal_result = False
        self._relaxed_miqp_model.set_instance(m1)
        self._relaxed_miqp_model.set_gurobi_param("MIPGap", 0)
        self._relaxed_miqp_model.set_gurobi_param("PoolSearchMode", 2)
        self._relaxed_miqp_model.set_gurobi_param("PoolSolutions", self.pool_size)
        for gurobi_option, value in self.relaxation_solver_options.items():
            if gurobi_option != "tee":
                self._relaxed_miqp_model.set_gurobi_param(gurobi_option, value)

        # Create a copy of the model, and save it for local solves
        LOGGER.info("Created a clone of the MINLP; this model is used for NLP solves.")
        self._nlp_model = get_gurobi(self.model.clone())
        self._nlp_model.config.raise_exception_on_nonoptimal_result = False
        self._nlp_model.set_gurobi_param("MIPGap", 0)
        self._nlp_model.set_gurobi_param("TimeLimit", self.time_limit_nlp_solves)
        for gurobi_option, value in self.nlp_solver_options.items():
            if gurobi_option != "tee":
                self._nlp_model.set_gurobi_param(gurobi_option, value)
        LOGGER.info("Solver object construction is complete.")

    @property
    def expression_pool(self):
        return pd.DataFrame(
            self._processed_expression_pool,
            index=[
                "Lower Bound",
                "Relaxation Time",
                "Expression",
                "NLP Status",
                "Upper Bound",
                "Total Time",
            ],
            columns=range(1, len(self._processed_expression_pool) + 1),
        ).transpose()

    @property
    def best_expression_tree(self):
        return self.expression_pool.sort_values(by="Upper Bound", ascending=True).iloc[
            0
        ]["Expression"]

    def _is_gap_closed(self):
        if self.best_bound >= self.best_incumbent:
            return True

        return (self.best_incumbent - self.best_bound <= EPS_A) or (
            self.best_incumbent - self.best_bound
            <= EPS_R * max(abs(self.best_bound), abs(self.best_incumbent))
        )

    def _extract_solution_pool(self):
        num_solutions = self._relaxed_miqp_model.get_model_attr("SolCount")
        # pylint: disable = protected-access
        pm = self._relaxed_miqp_model._pyomo_model
        gm_to_pm = self._relaxed_miqp_model._pyomo_var_to_solver_var_map

        for i in range(num_solutions):
            self._relaxed_miqp_model.set_gurobi_param("SolutionNumber", i)
            self.no_good_cuts.append(
                [
                    self._relaxed_miqp_model.get_model_attr("PoolNObjVal"),
                    self._relaxed_miqp_model.get_model_attr("Runtime"),
                    [
                        op
                        for op, v in pm.select_operator.items()
                        if gm_to_pm[v].PoolNX > 0.95
                    ],
                ]
            )

        self._relaxed_miqp_model.set_gurobi_param("SolutionNumber", 0)

    def _solve_relaxation(self, time_limit: float):
        # pylint: disable = protected-access
        # Add no good cuts from the previous solve
        LOGGER.debug("Solving the relaxed MIQCP model")
        pm = self._relaxed_miqp_model._pyomo_model  # Relaxed pyomo model
        while len(self.no_good_cuts) > 0:
            sol = self.no_good_cuts.pop()[-1]  # Get the last solution
            pm.no_good_cuts.add(
                sum(pm.select_operator[n, op] for n, op in sol) <= len(sol) - 1
            )

        LOGGER.debug("Finished adding no-good cuts")
        self._relaxed_miqp_model._solver_model.Params.TimeLimit = round(time_limit)
        self._relaxed_miqp_model.update()
        self._relaxed_miqp_model.solve(
            pm, tee=self.relaxation_solver_options.get("tee", False)
        )
        self._extract_solution_pool()

    def _solve_nlp(self, start_time: float):
        counter = len(self._processed_expression_pool)
        pm = self._nlp_model._pyomo_model

        for index, sol in enumerate(self.no_good_cuts):
            pm.select_operator.fix(0)
            for op in sol[-1]:
                pm.select_operator[op].fix(1)

            et = ExpressionTree(sol[-1])
            LOGGER.debug(f"Solving NLP for expression {et}")
            rc = et.get_redundant_constants(self.model.input_data_ref)

            # pylint: disable = logging-fstring-interpolation
            for cst_node_num in rc["mult_csts"]:
                LOGGER.warning(f"Fixing c_{cst_node_num} in {et} to 1")
                pm.constant_val[cst_node_num].fix(1)

            for cst_node_num in rc["add_csts"]:
                LOGGER.warning(f"Fixing c_{cst_node_num} in {et} to 0")
                pm.constant_val[cst_node_num].fix(0)

            try:
                self._nlp_model.update()
                result = self._nlp_model.solve(
                    pm, tee=self.nlp_solver_options.get("tee", False)
                )
                et.cst_values = {
                    n: pm.constant_val[n].value for n, op in sol[-1] if op == "cst"
                }
                summary = [
                    *sol[:2],
                    et,
                    str(result.solution_status).split(".", maxsplit=1)[-1],
                    result.incumbent_objective,
                    time.perf_counter() - start_time,
                ]
                self.best_bound = sol[0]
                self.best_incumbent = min(
                    self.best_incumbent, result.incumbent_objective
                )
            except NoFeasibleSolutionError:
                # self._nlp_model._solver_model.computeIIS()
                # self._nlp_model._solver_model.write(f"nlp_case_{counter + index}.ilp")
                summary = [
                    *sol[:2],
                    et,
                    "infeasible",
                    float("inf"),
                    time.perf_counter() - start_time,
                ]

            _solution_log(
                expression_num=counter + index + 1,
                lb=self.best_bound,
                current_ub=summary[4],
                solution_status=summary[3],
                best_ub=self.best_incumbent,
                comp_time=summary[5],
            )

            pm.constant_val.unfix()
            self._processed_expression_pool[counter + index] = summary
            if self._is_gap_closed():
                return True

            if self._has_time_limit_reached(start_time):
                return False

        return False

    def _has_time_limit_reached(self, start_time: float):
        if time.perf_counter() - start_time > self.overall_time_limit:
            return True

        return False

    def solve(self):
        """Solves the symbolic regression model"""
        found_global_optimum = False
        start_time = time.perf_counter()
        _solution_log(0, 0, 0, 0, 0, 0)
        while not found_global_optimum:
            if self._has_time_limit_reached(start_time):
                break

            self._solve_relaxation(
                time_limit=self.overall_time_limit - (time.perf_counter() - start_time)
            )
            found_global_optimum = self._solve_nlp(start_time)

        _solution_log(
            expression_num=-1 if found_global_optimum else -3,
            lb=self.best_bound,
            current_ub=0.0,
            solution_status="foo",
            best_ub=self.best_incumbent,
            comp_time=time.perf_counter() - start_time,
        )
