"""
Checks mccormick_envelopes on two Pyomo blocks.

Checks performed:
- builds a bounded example with x in [0, 6], y in [0, 4], and z in [5, 6];
- adds McCormick constraints for the bilinear term z = x*y on model.relaxation;
- checks that model.relaxation contains four McCormick constraints;
- checks that each McCormick constraint on model.relaxation is constructed;
- prints each McCormick constraint expression on model.relaxation;

- adds McCormick constraints for the ratio-style substitution x = z*y on model.div_relaxation;
- checks that model.div_relaxation contains four McCormick constraints;
- checks that each McCormick constraint on model.div_relaxation is constructed;
- prints each McCormick constraint expression on model.div_relaxation;
- prints the full Pyomo model with model.pprint().
"""

import pyomo.environ as pyo

from relaxations import mccormick_envelopes


def get_mccormick_constraints(blk):
    return [
        blk.mccormick_env_1,
        blk.mccormick_env_2,
        blk.mccormick_env_3,
        blk.mccormick_env_4,
    ]


def check_mccormick_constraints(blk):
    constraints = get_mccormick_constraints(blk)

    print("Created McCormick envelope on block:", blk.local_name)
    print("Number of constraints:", len(constraints))

    for constraint in constraints:
        print(f"{constraint.local_name}: {constraint.expr}")
        assert constraint.is_constructed()

    assert len(constraints) == 4


# Example in Grossmann's book, p.116
model = pyo.ConcreteModel()
model.x = pyo.Var(bounds=(0, 6))
model.y = pyo.Var(bounds=(0, 4))
model.z = pyo.Var(bounds=(5, 6))  # needed for convex relaxation of x/y
model.obj = pyo.Objective(expr=-model.x - model.y)
# model.bilinear_constraint = pyo.Constraint(expr=model.x * model.y <= 4)
model.substituted_bilinear = pyo.Constraint(expr=model.z <= 4)

# Block of convex relaxation for bilinear term, z = x*y
model.relaxation = pyo.Block()
mccormick_envelopes(
    model.relaxation,
    z=model.z,
    x=model.x,
    y=model.y,
)

check_mccormick_constraints(model.relaxation)

# Old return-based checks, used when mccormick_envelopes returned a ConstraintList:
# constraints = mccormick_envelopes(
#     model.relaxation,
#     z=model.z,
#     x=model.x,
#     y=model.y,
# )
# print("Created McCormick envelope:", constraints.local_name)
# print("Number of constraints:", len(constraints))
#
# for idx in constraints:
#     print(f"{constraints.local_name}[{idx}]: {constraints[idx].expr}")
#
# assert len(constraints) == 4

# Not in Grossmann's example, e.g. if there was an additional constraint x / y <= 5
# Block of convex relaxation for ratio, z = x/y => x = z*y
model.div_relaxation = pyo.Block()
mccormick_envelopes(
    model.div_relaxation,
    z=model.x,
    x=model.z,
    y=model.y,
)

check_mccormick_constraints(model.div_relaxation)

# Old return-based checks, used when mccormick_envelopes returned a ConstraintList:
# constraints = mccormick_envelopes(
#     model.div_relaxation,
#     z=model.x,
#     x=model.z,
#     y=model.y,
# )
# print("Created McCormick envelope:", constraints.local_name)
# print("Number of constraints:", len(constraints))
#
# for idx in constraints:
#     print(f"{constraints.local_name}[{idx}]: {constraints[idx].expr}")
#
# assert len(constraints) == 4


# Full model
model.pprint()
