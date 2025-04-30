"""Core modules package for the mudpy engine."""

# Copyright (c) 2004-2022 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

import importlib

import mudpy


def load():
    """Import/reload some modules (be careful, as this can result in loops)."""

    # dynamically build module list from package contents
    modules = []
    for module in mudpy.__spec__.loader.get_resource_reader().contents():

        if (
                # make sure it's a module file, not a directory
                module.endswith('.py')

                # don't include this file, we're inside it
                and module != '__init__.py'):

            # trim off the .py file extension
            modules.append(module[:-3])

    # iterate over the list of module files included in the package
    for module in modules:

        # attempt to reload the module, assuming it was probably imported
        # earlier
        try:
            importlib.reload(getattr(mudpy, module))

        # must not have been, so import it now
        except AttributeError:
            importlib.import_module("mudpy.%s" % module)


load()
