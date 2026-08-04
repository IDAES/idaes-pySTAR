"""
Checks _get_tangency_points with square and negative-square test functions.

Checks performed for the way the x-space is partitioned:
- "points" returns unchanged the input list of points;
- "uniform_x" with three points as input returns three equidistant points, including the bounds;
- "uniform_x" with one point returns the interval midpoint;
- "uniform_y" creates evenly spaced y-values and maps them back to x-space through func.inverse;
- "interval_bisection" for a convex function defined on [0,4] returns the x-coordinates (0, 1, 2, 3, 4);
- "interval_bisection" for a concave function on [0,4] returns the x-coordinates (0, 1, 2, 3, 4).
"""

import math

from relaxations import _get_tangency_points


class SquareFunction:
    def __call__(self, x):
        return x**2

    def inverse(self, y):
        return math.sqrt(y)


square = SquareFunction()


def square_derivative(x):
    return 2 * x


def negative_square(x):
    return -(x**2)


def negative_square_derivative(x):
    return -2 * x


# Explicit points are returned unchanged.
points = [0.25, 0.5, 0.75]
assert (
    _get_tangency_points(
        func=square,
        xlb=0,
        xub=1,
        num_points=3,
        points=points,
    )
    is points
)

# Uniform x-spacing includes the interval bounds.
assert _get_tangency_points(
    func=square,
    xlb=0,
    xub=4,
    num_points=3,
    points="uniform_x",
) == [0, 2, 4]

# With one tangent, uniform_x uses the midpoint.
assert _get_tangency_points(
    func=square,
    xlb=0,
    xub=4,
    num_points=1,
    points="uniform_x",
) == [2]

# Uniform y-spacing maps evenly spaced y-values back through func.inverse.
uniform_y_points = _get_tangency_points(
    func=square,
    xlb=1,
    xub=3,
    num_points=3,
    points="uniform_y",
)
expected_uniform_y_points = [1, math.sqrt(5), 3]
assert all(
    math.isclose(actual, expected)
    for actual, expected in zip(uniform_y_points, expected_uniform_y_points)
)

# Interval bisection starts with the bounds and midpoint, then bisects
# the interval with the largest current secant-to-tangent-envelope error.
assert _get_tangency_points(
    func=square,
    xlb=0,
    xub=4,
    num_points=5,
    points="interval_bisection",
    derivative=square_derivative,
    func_type="convex",
) == [0, 1, 2, 3, 4]

assert _get_tangency_points(
    func=negative_square,
    xlb=0,
    xub=4,
    num_points=5,
    points="interval_bisection",
    derivative=negative_square_derivative,
    func_type="concave",
) == [0, 1, 2, 3, 4]

print("All _get_tangency_points checks passed.")
