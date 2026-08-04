"""
Checks _outer_approximation_convex using f(x) = x**2 on [-1, 3].

Secant checks:
- the secant overestimator is constructed;
- it is tight at the interval endpoints x = -1 and x = 3;
- it is satisfied at x = 1 and at all sampled true-function points.

Tangent checks:
- the tangent underestimators are constructed;
- the tangency points are [-1, 1, 3];
- exactly three tangent constraints are generated;
- each tangent is tight at its own tangency point;
- every tangent is satisfied at all sampled true-function points.

Uniform-y checks for the square function (non-monotonic):
- y_interval=(0, 9) generates tangency points [0, sqrt(4.5), 3];
- each uniform-y tangent is tight at its own tangency point;
- every uniform-y tangent is satisfied at all sampled true-function points.
"""

import math
import pyomo.environ as pyo

from relaxations import _outer_approximation_convex


class SquareFunction:
    def __call__(self, x):
        return x**2

    def inverse(self, y):
        return math.sqrt(y)


square = SquareFunction()


def square_derivative(x):
    return 2 * x


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

_outer_approximation_convex(
    blk=model.relaxation,
    x=model.x,
    y=model.y,
    func=square,
    derivative=square_derivative,
    num_tangents=3,
    points="uniform_x",
    y_interval=None,
)

# Assert whether the secant and tangents are constructed
assert model.relaxation.linear_overestimator.is_constructed()
assert model.relaxation.linear_underestimators.is_constructed()

print("Secant overestimator:")
print(model.relaxation.linear_overestimator.expr)
print()

print("Tangent underestimators:")
for x0, constraint in model.relaxation.linear_underestimators.items():
    print(f"At x0 = {x0}: {constraint.expr}")
print()

# Assert whether the right number of tangents is constructed
tangency_points = list(model.relaxation.linear_underestimators.keys())
assert tangency_points == [-1, 1, 3]
assert len(model.relaxation.linear_underestimators) == 3

######## SECANT ############
# Assert that the secant is tight at the bounds of x
# The secant overestimator for x**2 on [-1, 3] is y <= 2*x + 3.
for x_value in (-1, 3):  # two x-coordinates of tangency points
    model.x.fix(x_value)
    model.y.fix(square(x_value))
    assert_constraint_tight(model.relaxation.linear_overestimator)

# Take a feasible point and check whether it satisfies the secant constraint
model.x.fix(1)
model.y.fix(square(1))
assert_constraint_satisfied(model.relaxation.linear_overestimator)


######### TANGENTS #########
# Each tangent underestimator is tight at its own tangency point.
for x0 in tangency_points:
    model.x.fix(x0)
    model.y.fix(square(x0))
    assert_constraint_tight(model.relaxation.linear_underestimators[x0])

# The exact convex function should satisfy every generated relaxation line.
for x_value in (-1, 0, 1, 2, 3):  # sampled x-coordinates
    model.x.fix(x_value)
    model.y.fix(square(x_value))
    assert_constraint_satisfied(model.relaxation.linear_overestimator)

    for constraint in model.relaxation.linear_underestimators.values():
        assert_constraint_satisfied(constraint)


###### Uniform-y checks for the square function (non-monotonic)
# y_interval is specified

uniform_y_model = pyo.ConcreteModel()
uniform_y_model.x = pyo.Var(bounds=(-1, 3))
uniform_y_model.y = pyo.Var()
uniform_y_model.relaxation = pyo.Block()

_outer_approximation_convex(
    blk=uniform_y_model.relaxation,
    x=uniform_y_model.x,
    y=uniform_y_model.y,
    func=square,
    derivative=square_derivative,
    num_tangents=3,
    points="uniform_y",
    y_interval=(0, 9),
)

print("Uniform-y tangent underestimators:")
for x0, constraint in uniform_y_model.relaxation.linear_underestimators.items():
    print(f"At x0 = {x0}: {constraint.expr}")
print()

uniform_y_tangency_points = list(
    uniform_y_model.relaxation.linear_underestimators.keys()
)
expected_uniform_y_tangency_points = [0, math.sqrt(4.5), 3]
assert len(uniform_y_model.relaxation.linear_underestimators) == 3
assert all(
    math.isclose(actual, expected)
    for actual, expected in zip(
        uniform_y_tangency_points, expected_uniform_y_tangency_points
    )
)

for x0 in uniform_y_tangency_points:
    uniform_y_model.x.fix(x0)
    uniform_y_model.y.fix(square(x0))
    assert_constraint_tight(uniform_y_model.relaxation.linear_underestimators[x0])

for x_value in (-1, 0, 1, 2, 3):
    uniform_y_model.x.fix(x_value)
    uniform_y_model.y.fix(square(x_value))
    assert_constraint_satisfied(uniform_y_model.relaxation.linear_overestimator)

    for constraint in uniform_y_model.relaxation.linear_underestimators.values():
        assert_constraint_satisfied(constraint)

print("All _outer_approximation_convex checks passed.")
