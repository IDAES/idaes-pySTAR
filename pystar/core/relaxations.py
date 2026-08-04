from typing import Callable, Optional
import pyomo.environ as pyo


def mccormick_envelopes(
    blk: pyo.Block,
    z: pyo.Var,
    x: pyo.Var,
    y: pyo.Var,
    disjunct_var: pyo.Var | None = None,
):
    # z = x * y
    xlb, xub = x.lb, x.ub
    ylb, yub = y.lb, y.ub
    zlb, zub = (
        z.lb,
        z.ub,
    )  # needed for modeling convex relaxation for z = x/y => x = z*y

    if xlb is None or xub is None:
        raise ValueError(
            "Variable x should have lower and and upper bounds to construct McCormick relaxation"
        )

    if ylb is None or yub is None:
        raise ValueError(
            "Variable y should have lower and and upper bounds to construct McCormick relaxation"
        )

    if disjunct_var is None:
        disjunct_var = 1

    # The following constraints are added to the input pyo.Block
    # Counter needed if mccormick_envelopes is called multiple times on the same block
    # if not hasattr(blk, "_mccormick_counter"):
    #    blk._mccormick_counter = 0

    # name = f"mccormick_{blk._mccormick_counter}"
    # blk._mccormick_counter += 1

    # constraints = pyo.ConstraintList()
    # blk.add_component(name, constraints)
    # constraints.add(z >= x * ylb + xlb * y - xlb * ylb)
    # constraints.add(z >= xub * y + x * yub - xub * yub)
    # constraints.add(z <= x * yub + xlb * y - xlb * yub)
    # constraints.add(z <= x * ylb + xub * y - xub * ylb)
    # return constraints

    # Linear underestimators
    blk.mccormick_env_1 = pyo.Constraint(
        expr=z >= x * ylb + xlb * y - xlb * ylb * disjunct_var
    )
    blk.mccormick_env_2 = pyo.Constraint(
        expr=z >= xub * y + x * yub - xub * yub * disjunct_var
    )

    # Linear overestimators
    blk.mccormick_env_3 = pyo.Constraint(
        expr=z <= x * yub + xlb * y - xlb * yub * disjunct_var
    )
    blk.mccormick_env_4 = pyo.Constraint(
        expr=z <= x * ylb + xub * y - xub * ylb * disjunct_var
    )

    # return [
    #    z >= x * ylb + xlb * y - xlb * ylb,
    #    z >= xub * y + x * yub - xub * yub,
    #    z <= x * yub + xlb * y - xlb * yub,
    #    z <= x * ylb + xub * y - xub * ylb,
    # ]


### Outer Approximation
# Returns the range of convex/concave functions (to be used when generating tangents with uniform y-spacing)
def _compute_function_interval(func: Callable, xlb: float, xub: float):
    return min(func(xlb), func(xub)), max(
        func(xlb), func(xub)
    )  # works only for monotonic functions (e.g. doesn't work for x^2 in [-1,1])


# Returns the list of tangency points
def _get_tangency_points(
    func: Callable,
    xlb: float,
    xub: float,
    num_points: int,
    points: str | list,  # str (e.g. "uniform_x") or specific list of points
    derivative: Optional[Callable] = None,
    inverse: Optional[Callable] = None,
    func_type: Optional[str] = None,
    y_interval: Optional[tuple] = None,
):

    # Case where points is a list
    if isinstance(points, list):  # type(points)==list is false for subclass of list
        return points

    # Case where points is a string
    if points == "uniform_x":

        if num_points == 1:
            return [(xlb + xub) / 2]  # the middle point (one-element list)

        # If num_points greater than 1:
        x_step = (xub - xlb) / (num_points - 1)
        x_coordinates_list = [xlb + i * x_step for i in range(num_points)]
        return x_coordinates_list

    if points == "uniform_y":
        if y_interval is None:
            ylb, yub = _compute_function_interval(func, xlb, xub)
        else:
            ylb, yub = y_interval

        if num_points == 1:
            y_mid = (ylb + yub) / 2
            return [
                inverse(y_mid)
            ]  # the x-image of the y middle point (one-element list)

        y_step = (yub - ylb) / (num_points - 1)
        y_coordinates_list = [ylb + i * y_step for i in range(num_points)]
        x_coordinates_list = [inverse(y) for y in y_coordinates_list]
        return x_coordinates_list

    if points == "interval_bisection":

        def tangent_at(x0):
            tangent_slope = derivative(x0)
            tangent_intercept = func(x0) - tangent_slope * x0
            return tangent_slope, tangent_intercept

        def evaluate_line(line, x):  # line is a 2-tuple (slope, intercept)
            slope, intercept = line
            return slope * x + intercept

        secant_slope = (func(xub) - func(xlb)) / (xub - xlb)
        secant_intercept = func(xlb) - secant_slope * xlb
        secant = (secant_slope, secant_intercept)

        first_x = (xlb + xub) / 2
        x_coordinates_list = [xlb, first_x, xub]

        intervals = [
            (xlb, first_x),
            (first_x, xub),
        ]

        while len(x_coordinates_list) < num_points:
            tangents = [tangent_at(x0) for x0 in x_coordinates_list]

            def tangent_envelope(x):
                tangent_values = [evaluate_line(tangent, x) for tangent in tangents]

                if func_type == "convex":
                    return max(tangent_values)

                if func_type == "concave":
                    return min(tangent_values)

            def interval_error(interval):
                a, b = interval  # 2-tuple representing interval [a,b]
                midpoint = (a + b) / 2

                envelope_value = tangent_envelope(midpoint)
                secant_value = evaluate_line(secant, midpoint)

                if func_type == "convex":
                    return secant_value - envelope_value

                if func_type == "concave":
                    return envelope_value - secant_value

            worst_interval = max(intervals, key=interval_error)

            a, b = worst_interval
            new_x = (a + b) / 2

            x_coordinates_list.append(new_x)

            intervals.remove(worst_interval)
            intervals.append((a, new_x))
            intervals.append((new_x, b))

        return sorted(x_coordinates_list)

    raise ValueError(
        f"Unrecognized strategy for construncting points of tangency: {points} "
    )


def outer_approximation(
    blk: pyo.Block,
    x: pyo.Var,
    y: pyo.Var,
    func: Callable,
    derivative: Callable,
    inverse: Callable,
    func_type: str,
    num_tangents: int = 5,
    points: str | list = "uniform_x",
    y_interval: Optional[tuple] = None,
    disjunct_var: pyo.Var | None = None,
):
    if func_type == "convex":
        _outer_approximation_convex(
            blk,
            x,
            y,
            func,
            derivative,
            inverse,
            num_tangents,
            points,
            y_interval,
            disjunct_var,
        )
    elif func_type == "concave":
        _outer_approximation_concave(
            blk,
            x,
            y,
            func,
            derivative,
            inverse,
            num_tangents,
            points,
            y_interval,
            disjunct_var,
        )
    else:
        raise ValueError("Function must either be convex or concave")


def _outer_approximation_convex(
    blk: pyo.Block,
    x: pyo.Var,
    y: pyo.Var,
    func: Callable,
    derivative: Callable,
    inverse: Callable,
    num_tangents: int,
    points: str | list,
    y_interval: Optional[tuple],
    disjunct_var: pyo.Var | None = None,
):
    xlb, xub = x.lb, x.ub

    if disjunct_var is None:
        disjunct_var = 1

    # Secant as linear overestimator
    secant_slope = (func(xub) - func(xlb)) / (xub - xlb)
    blk.linear_overestimator = pyo.Constraint(
        expr=y <= secant_slope * (x - xlb) + func(xlb) * disjunct_var
    )

    # A number of tangents as linear underestimators
    tangency_points = _get_tangency_points(
        func=func,
        xlb=xlb,
        xub=xub,
        num_points=num_tangents,
        points=points,
        y_interval=y_interval,
        inverse=inverse,
        derivative=derivative,
    )

    @blk.Constraint(tangency_points)
    def linear_underestimators(_, x0):
        tangent_slope = derivative(x0)
        tangent_intercept = func(x0) - tangent_slope * x0
        return y >= tangent_slope * x + tangent_intercept * disjunct_var


def _outer_approximation_concave(
    blk: pyo.Block,
    x: pyo.Var,
    y: pyo.Var,
    func: Callable,
    derivative: Callable,
    inverse: Callable,
    num_tangents: int,
    points: str | list,
    y_interval: Optional[tuple],
    disjunct_var: pyo.Var | None = None,
):

    xlb, xub = x.lb, x.ub

    if disjunct_var is None:
        disjunct_var = 1

    # Secant as linear underestimator
    secant_slope = (func(xub) - func(xlb)) / (xub - xlb)
    blk.linear_underestimator = pyo.Constraint(
        expr=y >= secant_slope * (x - xlb) + func(xlb) * disjunct_var
    )

    # A number of tangents as linear overestimators
    tangency_points = _get_tangency_points(
        func=func,
        xlb=xlb,
        xub=xub,
        num_points=num_tangents,
        points=points,
        y_interval=y_interval,
    )

    @blk.Constraint(tangency_points)
    def linear_overestimators(_, x0):
        tangent_slope = derivative(x0)
        tangent_intercept = func(x0) - tangent_slope * x0
        return y <= tangent_slope * x + tangent_intercept * disjunct_var
