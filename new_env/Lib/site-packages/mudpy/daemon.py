# Copyright (c) 2004-2018 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

# core objects for the mudpy engine
import importlib
import sys

import mudpy


def main():

    # start it up
    mudpy.misc.setup()

    # loop indefinitely while the world is not flagged for termination or
    # there are still connected users
    while (not mudpy.misc.universe.terminate_flag or
           mudpy.misc.universe.userlist):

        # the world was flagged for a reload of all code/data
        if mudpy.misc.universe.reload_flag:
            importlib.reload(mudpy)
            mudpy.misc.reload_data()
            mudpy.misc.universe.reload_flag = False

        # do what needs to be done on each pulse
        mudpy.misc.on_pulse()

    # shut it all down
    mudpy.misc.finish()


if __name__ == '__main__':
    sys.exit(main())
