#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

"""
This module modifies the declare_custom_block decorator available in
pyomo.core.base.block to set a default rule argument.
"""

# Installed libs
import pyomo.environ as pyo
import pytest

# User-defined libs
from primo.custom_block.custom_block import BlockData, declare_custom_block


# pylint: disable = undefined-variable, attribute-defined-outside-init
# pylint: disable = missing-function-docstring
def test_custom_block_1():
    """Tests the decorator without the `build` method"""

    @declare_custom_block("FooBlock")
    class FooBlockData(BlockData):
        """Construct an empty custom block"""

    m = pyo.ConcreteModel()
    m.blk_without_index = FooBlock()
    m.blk_with_index = FooBlock([1, 2, 3, 4])

    assert isinstance(m.blk_without_index, FooBlockData)
    for p in m.blk_with_index:
        assert isinstance(m.blk_with_index[p], FooBlockData)
    assert len(m.blk_with_index) == 4


def test_custom_block_2():
    """Tests the decorator with the `build` method, but without options"""

    @declare_custom_block("FooBlock")
    class FooBlockData(BlockData):
        """A dummy custom block"""

        def build(self, *args):
            self.x = pyo.Var(list(args))
            self.y = pyo.Var()

    m = pyo.ConcreteModel()
    m.blk_without_index = FooBlock()
    m.blk_1 = FooBlock([1, 2, 3])
    m.blk_2 = FooBlock([4, 5], [6, 7])

    assert isinstance(m.blk_without_index, FooBlockData)
    for p in m.blk_1:
        assert isinstance(m.blk_1[p], FooBlockData)

    for p in m.blk_2:
        assert isinstance(m.blk_2[p], FooBlockData)

    assert hasattr(m.blk_without_index, "x")
    assert hasattr(m.blk_without_index, "y")

    assert len(m.blk_1) == 3
    assert len(m.blk_2) == 4

    assert len(m.blk_1[2].x) == 1
    assert len(m.blk_2[4, 6].x) == 2


def test_custom_block_3():
    """Tests the decorator with the `build` method and model options"""

    @declare_custom_block("FooBlock")
    class FooBlockData(BlockData):
        """A dummy custom block"""

        def build(self, *args, capex, opex):
            self.x = pyo.Var(list(args))
            self.y = pyo.Var()

            self.capex = capex
            self.opex = opex

    options = {"capex": 42, "opex": 24}

    m = pyo.ConcreteModel()
    m.blk_without_index = FooBlock(model_options=options)
    m.blk_1 = FooBlock([1, 2, 3], model_options=options)
    m.blk_2 = FooBlock([4, 5], [6, 7], model_options=options)

    assert isinstance(m.blk_without_index, FooBlockData)
    for p in m.blk_1:
        assert isinstance(m.blk_1[p], FooBlockData)

    for p in m.blk_2:
        assert isinstance(m.blk_2[p], FooBlockData)

    assert hasattr(m.blk_without_index, "x")
    assert hasattr(m.blk_without_index, "y")
    assert m.blk_without_index.capex == 42
    assert m.blk_without_index.opex == 24

    assert len(m.blk_1) == 3
    assert len(m.blk_2) == 4

    assert len(m.blk_1[2].x) == 1
    assert len(m.blk_2[4, 6].x) == 2

    assert m.blk_1[3].capex == 42
    assert m.blk_2[4, 7].opex == 24

    with pytest.raises(TypeError):
        # missing 2 required keyword-only arguments
        m.blk_3 = FooBlock()


def test_custom_block_4():
    """Tests if the default rule can be overwritten"""

    @declare_custom_block("FooBlock")
    class FooBlockData(BlockData):
        """A dummy custom block"""

        def build(self, *args):
            self.x = pyo.Var(list(args))
            self.y = pyo.Var()

    def _new_rule(blk):
        blk.a = pyo.Var()
        blk.b = pyo.Var()

    m = pyo.ConcreteModel()
    m.blk = FooBlock(rule=_new_rule)

    assert isinstance(m.blk, FooBlockData)
    assert not hasattr(m.blk, "x")
    assert not hasattr(m.blk, "y")
    assert isinstance(m.blk.a, pyo.Var)
    assert isinstance(m.blk.b, pyo.Var)
