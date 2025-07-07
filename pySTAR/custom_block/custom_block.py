#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
This module modifies the declare_custom_block decorator available in
pyomo.core.base.block to set a default rule argument.
"""

# Standard libs
import sys

# Installed libs
# pylint: disable = unused-import
# Although BlockData is not needed for this file, we are importing it so that
# both BlockData and declare_custom_block can be imported from this file when needed.
# Otherwise, BlockData needs to be imported separately from Pyomo.
from pyomo.core.base.block import (
    Block,
    BlockData,
    ScalarCustomBlockMixin,
    UnindexedComponent_set,
)


def _default_rule(model_options):
    """
    Default rule for custom blocks

    Parameters
    ----------
    model_options : dict
        Dictionary of options needed to construct the block model
    """

    def _rule(blk, *args):
        try:
            # Attempt to build the model
            blk.build(*args, **model_options)

        except AttributeError:
            # build method is not implemented in the BlockData class
            # Returning an empty Pyomo Block
            pass

    return _rule


# pylint: disable = protected-access, no-member
class CustomBlock(Block):
    """The base class used by instances of custom block components"""

    def __init__(self, *args, **kwargs):
        config = {key: kwargs.pop(key) for key in self._build_options if key in kwargs}
        kwargs.setdefault("rule", _default_rule(config))

        if self._default_ctype is not None:
            kwargs.setdefault("ctype", self._default_ctype)
        Block.__init__(self, *args, **kwargs)

    def __new__(cls, *args, **kwargs):
        if cls.__bases__[0] is not CustomBlock:
            # we are creating a class other than the "generic" derived
            # custom block class.  We can assume that the routing of the
            # generic block class to the specific Scalar or Indexed
            # subclass has already occurred and we can pass control up
            # to (toward) object.__new__()
            return super().__new__(cls, *args, **kwargs)
        # If the first base class is this CustomBlock class, then the
        # user is attempting to create the "generic" block class.
        # Depending on the arguments, we need to map this to either the
        # Scalar or Indexed block subclass.
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return super().__new__(cls._scalar_custom_block, *args, **kwargs)
        else:
            return super().__new__(cls._indexed_custom_block, *args, **kwargs)


def declare_custom_block(name, model_options=None):
    """Decorator to declare components for a custom block data class

    >>> @declare_custom_block(name="FooBlock")
    ... class FooBlockData(BlockData):
    ...    # custom block data class
    ...    def build(self, *args, **kwargs):  # Default rule argument
    ...        pass
    """

    def block_data_decorator(block_data):
        # this is the decorator function that creates the block
        # component classes

        # Declare the new Block component (derived from CustomBlock)
        # corresponding to the BlockData that we are decorating
        #
        # Note the use of `type(CustomBlock)` to pick up the metaclass
        # that was used to create the CustomBlock (in general, it should
        # be `type`)
        comp = type(CustomBlock)(
            name,  # name of new class
            (CustomBlock,),  # base classes
            # class body definitions (populate the new class' __dict__)
            {
                # ensure the created class is associated with the calling module
                "__module__": block_data.__module__,
                # Default IndexedComponent data object is the decorated class:
                "_ComponentDataClass": block_data,
                # By default this new block does not declare a new ctype
                "_default_ctype": None,
                "_build_options": [] if model_options is None else model_options,
            },
        )

        # Declare Indexed and Scalar versions of the custom block.  We
        # will register them both with the calling module scope, and
        # with the CustomBlock (so that CustomBlock.__new__ can route
        # the object creation to the correct class)
        comp._indexed_custom_block = type(comp)(
            "Indexed" + name,
            (comp,),
            {  # ensure the created class is associated with the calling module
                "__module__": block_data.__module__
            },
        )
        comp._scalar_custom_block = type(comp)(
            "Scalar" + name,
            (ScalarCustomBlockMixin, block_data, comp),
            {  # ensure the created class is associated with the calling module
                "__module__": block_data.__module__
            },
        )

        # Register the new Block types in the same module as the BlockData
        for _cls in (comp, comp._indexed_custom_block, comp._scalar_custom_block):
            setattr(sys.modules[block_data.__module__], _cls.__name__, _cls)
        return block_data

    return block_data_decorator
