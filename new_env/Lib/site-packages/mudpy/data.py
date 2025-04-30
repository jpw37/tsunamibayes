"""Data interface functions for the mudpy engine."""

# Copyright (c) 2004-2022 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

import os
import re
import stat

import mudpy
import yaml


class _IBSEmitter(yaml.emitter.Emitter):

    """Override the default YAML Emitter to indent block sequences."""

    def expect_block_sequence(self):
        """Match the expectations of the ``yamllint`` style checker."""

        # TODO(fungi) Get an option for this implemented upstream in
        # the pyyaml library
        self.increase_indent(flow=False, indentless=False)
        self.state = self.expect_first_block_sequence_item


class _IBSDumper(yaml.SafeDumper, _IBSEmitter):

    """Use our _IBSEmitter instead of the default implementation."""

    pass


class Data:

    """A file containing universe elements and their facets."""

    def __init__(self,
                 source,
                 universe,
                 flags=None,
                 relative=None,
                 ):
        self.source = source
        self.universe = universe
        if flags is None:
            self.flags = []
        else:
            self.flags = flags[:]
        self.relative = relative
        self.load()

    def load(self):
        """Read a file, create elements and poplulate facets accordingly."""
        self.modified = False
        self.source = find_file(
                self.source, relative=self.relative, universe=self.universe)
        try:
            with open(self.source) as datafd:
                self.data = yaml.safe_load(datafd)
            log_entry = ("Loaded file %s into memory." % self.source, 5)
        except FileNotFoundError:
            # it's normal if the file is one which doesn't exist yet
            self.data = {}
            log_entry = (
                "File %s was not found and will be created." % self.source, 6)
        try:
            mudpy.misc.log(*log_entry)
        except NameError:
            # happens when we're not far enough along in the init process
            self.universe.setup_loglines.append(log_entry)
        if not hasattr(self.universe, "files"):
            self.universe.files = {}
        self.universe.files[self.source] = self
        includes = []
        for node in list(self.data):
            if node == "_load":
                includes += self.data["_load"]
                continue
            if node.startswith("_"):
                continue
            facet_pos = node.rfind(".") + 1
            prefix = node[:facet_pos].strip(".")
            try:
                element = self.universe.contents[prefix]
            except KeyError:
                element = mudpy.misc.Element(prefix, self.universe, self)
            element.set(node[facet_pos:], self.data[node])
            if prefix.startswith("mudpy.movement."):
                self.universe.directions.add(
                    prefix[prefix.rfind(".") + 1:])
        for include_file in includes:
            if not os.path.isabs(include_file):
                include_file = find_file(
                    include_file,
                    relative=self.source,
                    universe=self.universe
                )
            if (include_file not in self.universe.files or not
                    self.universe.files[include_file].is_writeable()):
                Data(include_file, self.universe)

    def save(self):
        """Write the data, if necessary."""
        normal_umask = 0o0022
        private_umask = 0o0077
        private_file_mode = 0o0600

        # when modified, writeable and has content or the file exists
        if self.modified and self.is_writeable() and (
           self.data or os.path.exists(self.source)
           ):

            # make parent directories if necessary
            old_umask = os.umask(normal_umask)
            os.makedirs(os.path.dirname(self.source), exist_ok=True)
            os.umask(old_umask)

            # backup the file
            if "mudpy.limit" in self.universe.contents:
                max_count = self.universe.contents["mudpy.limit"].get(
                    "backups", 0)
            else:
                max_count = 0
            if os.path.exists(self.source) and max_count:
                backups = []
                for candidate in os.listdir(os.path.dirname(self.source)):
                    if re.match(
                       os.path.basename(self.source) +
                       r"""\.\d+$""", candidate
                       ):
                        backups.append(int(candidate.split(".")[-1]))
                backups.sort()
                backups.reverse()
                for old_backup in backups:
                    if old_backup >= max_count - 1:
                        os.remove(self.source + "." + str(old_backup))
                    elif not os.path.exists(
                        self.source + "." + str(old_backup + 1)
                    ):
                        os.rename(
                            self.source + "." + str(old_backup),
                            self.source + "." + str(old_backup + 1)
                        )
                if not os.path.exists(self.source + ".0"):
                    os.rename(self.source, self.source + ".0")

            # our data file
            if "private" in self.flags:
                old_umask = os.umask(private_umask)
                file_descriptor = open(self.source, "w")
                if oct(stat.S_IMODE(os.stat(
                        self.source)[stat.ST_MODE])) != private_file_mode:
                    # if it's marked private, chmod it appropriately
                    os.chmod(self.source, private_file_mode)
            else:
                old_umask = os.umask(normal_umask)
                file_descriptor = open(self.source, "w")
            os.umask(old_umask)

            # write and close the file
            yaml.dump(self.data, Dumper=_IBSDumper, allow_unicode=True,
                      default_flow_style=False, explicit_start=True, indent=4,
                      stream=file_descriptor)
            file_descriptor.close()

            # unset the modified flag
            self.modified = False

    def is_writeable(self):
        """Returns True if the _lock is False."""
        try:
            return not self.data.get("_lock", False)
        except KeyError:
            return True


def find_file(
    file_name=None,
    group=None,
    prefix=None,
    relative=None,
    search=None,
    stash=None,
    universe=None
):
    """Return an absolute file path based on configuration."""

    # this is all unnecessary if it's already absolute
    if file_name and os.path.isabs(file_name):
        return os.path.realpath(file_name)

    # if a universe was provided, try to get some defaults from there
    if universe:

        if hasattr(
                universe, "contents") and "mudpy.filing" in universe.contents:
            filing = universe.contents["mudpy.filing"]
            if not prefix:
                prefix = filing.get("prefix")
            if not search:
                search = filing.get("search")
            if not stash:
                stash = filing.get("stash")

        # if there's only one file loaded, try to work around a chicken<egg
        elif hasattr(universe, "files") and len(
            universe.files
        ) == 1 and not universe.files[
                list(universe.files.keys())[0]].is_writeable():
            data_file = universe.files[list(universe.files.keys())[0]].data

            # try for a fallback default directory
            if not stash:
                stash = data_file.get(".mudpy.filing.stash", "")

            # try for a fallback root path
            if not prefix:
                prefix = data_file.get(".mudpy.filing.prefix", "")

            # try for a fallback search path
            if not search:
                search = data_file.get(".mudpy.filing.search", "")

        # another fallback root path, this time from the universe startdir
        if hasattr(universe, "startdir"):
            if not prefix:
                prefix = universe.startdir
            elif not os.path.isabs(prefix):
                prefix = os.path.join(universe.startdir, prefix)

    # when no root path is specified, assume the current working directory
    if (not prefix or prefix == ".") and hasattr(universe, "startdir"):
        prefix = universe.startdir

    # make sure it's absolute
    prefix = os.path.realpath(prefix)

    # if there's no search path, just use the root path and etc
    if not search:
        search = [prefix, "etc"]

    # work on a copy of the search path, to avoid modifying the caller's
    else:
        search = search[:]

    # if there's no default path, use the last component of the search path
    if not stash:
        stash = search[-1]

    # if an existing file or directory reference was supplied, prepend it
    if relative:
        if os.path.isdir(relative):
            search = [relative] + search
        else:
            search = [os.path.dirname(relative)] + search

    # make the search path entries absolute and throw away any dupes
    clean_search = []
    for each_path in search:
        if not os.path.isabs(each_path):
            each_path = os.path.realpath(os.path.join(prefix, each_path))
        if each_path not in clean_search:
            clean_search.append(each_path)

    # start hunting for the file now
    for each_path in clean_search:

        # construct the candidate path
        candidate = os.path.join(each_path, file_name)

        # if the file exists and is readable, we're done
        if os.path.isfile(candidate):
            file_name = os.path.realpath(candidate)
            break

        # if the path is a directory, look for an __init__ file
        if os.path.isdir(candidate):
            file_name = os.path.realpath(
                    os.path.join(candidate, "__init__.yaml"))
            break

    # it didn't exist after all, so use the default path instead
    if not os.path.isabs(file_name):
        file_name = os.path.join(stash, file_name)
    if not os.path.isabs(file_name):
        file_name = os.path.join(prefix, file_name)

    # and normalize it last thing before returning
    file_name = os.path.realpath(file_name)
    return file_name
