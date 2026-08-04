"""
Checks _outer_approximation_concave using f(x) = -x**2 on [-1, 3].

Secant checks:
- the secant underestimator is constructed;
- it is tight at the interval endpoints x = -1 and x = 3;
- it is satisfied at x = 1 and at all sampled true-function points.

Tangent checks:
- the tangent overestimators are constructed;
- the tangency points are [-1, 1, 3];
- exactly three tangent constraints are generated;
- each tangent is tight at its own tangency point;
- every tangent is satisfied at all sampled true-function points.
"""

import math
import pyomo.environ as pyo

from relaxations import _outer_approximation_concave


def negative_square(x):
    return -(x**2)


def negative_square_derivative(x):
    return -2 * x


# The function checks: after plugging in the current values of x, y of a feasible point in the given secant or tangent, is the constraint satisfied?
def assert_constraint_satisfied(constraint, tol=1e-9):
    body_value = pyo.value(constraint.body)

    if constraint.has_lb():
        assert body_value >= pyo.value(constraint.lower) - tol

    if constraint.has_ub():
        assert body_value <= pyo.value(constraint.upper) + tol


# The function checks: after plugging in the current values of x, y of an endpoint in the given secant or tangent, is the constraint tight, i.e. does the point lie exactly on the constraint line?
#
def assert_constraint_tight(constraint, tol=1e-9):
    body_value = pyo.value(constraint.body)

    if constraint.has_lb() and math.isclose(
        body_value, pyo.value(constraint.lower), abs_tol=tol
    ):
        return

    if constraint.has_ub() and math.isclose(
        body_value, pyo.value(constraint.upper), abs_tol=tol
    ):
        return

    raise AssertionError(f"Constraint is not tight: {constraint.expr}")


model = pyo.ConcreteModel()
model.x = pyo.Var(bounds=(-1, 3))
model.y = pyo.Var()
model.relaxation = pyo.Block()

_outer_approximation_concave(
    blk=model.relaxation,
    x=model.x,
    y=model.y,
    func=negative_square,
    derivative=negative_square_derivative,
    num_tangents=3,
    points="uniform_x",
    y_interval=None,
)

# Assert whether the secant and tangents are constructed
assert model.relaxation.linear_underestimator.is_constructed()
assert model.relaxation.linear_overestimators.is_constructed()

print("Secant underestimator:")
print(model.relaxation.linear_underestimator.expr)
print()

print("Tangent overestimators:")
for x0, constraint in model.relaxation.linear_overestimators.items():
    print(f"At x0 = {x0}: {constraint.expr}")
print()

# Assert whether the right number of tangents is constructed
tangency_points = list(model.relaxation.linear_overestimators.keys())
assert tangency_points == [-1, 1, 3]
assert len(model.relaxation.linear_overestimators) == 3

######## SECANT ############
# Assert that the secant is tight at the bounds of x
# The secant underestimator for -x**2 on [-1, 3] is y >= -2*x - 3.
for x_value in (-1, 3):  # two x-coordinates of tangency points
    model.x.fix(x_value)
    model.y.fix(negative_square(x_value))
    assert_constraint_tight(model.relaxation.linear_underestimator)

# Take a feasible point and check whether it satisfies the secant constraint
model.x.fix(1)
model.y.fix(negative_square(1))
assert_constraint_satisfied(model.relaxation.linear_underestimator)


######### TANGENTS #########
# Each tangent overestimator is tight at its own tangency point.
for x0 in tangency_points:
    model.x.fix(x0)
    model.y.fix(negative_square(x0))
    assert_constraint_tight(model.relaxation.linear_overestimators[x0])

# The exact concave function should satisfy every generated relaxation line.
for x_value in (-1, 0, 1, 2, 3):  # sampled x-coordinates
    model.x.fix(x_value)
    model.y.fix(negative_square(x_value))
    assert_constraint_satisfied(model.relaxation.linear_underestimator)

    for constraint in model.relaxation.linear_overestimators.values():
        assert_constraint_satisfied(constraint)

print("All _outer_approximation_concave checks passed.")
