"""Version and diagnostic information for the mudpy engine."""

# Copyright (c) 2018-2021 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

import json
import sys


# TODO(fungi) Clean up once Python 3.8 is the oldest interpreter supported
try:
    import importlib.metadata
    use_importlib = True
except ModuleNotFoundError:
    import pkg_resources
    use_importlib = False


class VersionDetail:

    """Version detail for a Python package."""

    def __init__(self, package):
        if use_importlib:
            project_name = package.metadata.get('Name')
        else:
            project_name = package.project_name
        self.project_name = _normalize_project(project_name)
        version = package.version
        self.version_info = tuple(version.split('.'))

        # Build up a human-friendly version string for display purposes
        self.text = "%s %s" % (self.project_name, version)

        # Obtain Git commit ID from PBR metadata if present
        if use_importlib:
            dist = importlib.metadata.distribution(self.project_name)
        else:
            dist = pkg_resources.get_distribution(self.project_name)
        try:
            if use_importlib:
                pbr_metadata = dist.read_text("pbr.json")
            else:
                pbr_metadata = dist.get_metadata("pbr.json")
        except (IOError, KeyError):
            pbr_metadata = None
        if pbr_metadata:
            self.git_version = json.loads(pbr_metadata)["git_version"]
        else:
            self.git_version = None
        if self.git_version:
            self.text = "%s (%s)" % (self.text, self.git_version)

    def __repr__(self):
        return self.text


class Versions:

    """Tracks info on known Python package versions."""

    def __init__(self, project_name):
        # Normalize the supplied name
        project_name = _normalize_project(project_name)

        # Python info for convenience
        self.python_version = "%s Python %s" % (
            sys.platform, sys.version.split(" ")[0])

        # List of package names for this package's declared dependencies
        requirements = []
        if use_importlib:
            for req in importlib.metadata.distribution(project_name).requires:
                requirements.append(_normalize_project(req))
        else:
            for req in pkg_resources.get_distribution(project_name).requires():
                requirements.append(_normalize_project(req.project_name))

        # Accumulators for Python package versions
        self.dependencies = {}
        self.environment = {}

        # Loop over all installed packages
        if use_importlib:
            distributions = importlib.metadata.distributions()
        else:
            distributions = pkg_resources.working_set
        for package in distributions:
            version = VersionDetail(package)
            # Sort packages into the corresponding buckets
            if version.project_name in requirements:
                # This is a dependency
                self.dependencies[version.project_name] = version
            elif version.project_name == project_name:
                # This is our main package
                self.version = version
            else:
                # This may be a transitive dep, or merely installed
                self.environment[version.project_name] = version

        self.dependencies_text = ", ".join(
            sorted([x.text for x in self.dependencies.values()]))
        self.environment_text = ", ".join(
            sorted([x.text for x in self.environment.values()]))

    def __repr__(self):
        return "Running %s on %s with dependencies %s." % (
            self.version.text,
            self.python_version,
            self.dependencies_text,
            )


def _normalize_project(project_name):
    """Strip and normalize Python project names."""

    # Use lower-case names for ease of comparison
    if use_importlib:
        project_name = project_name.lower()
    else:
        project_name = pkg_resources.safe_name(project_name).lower()

    # Remove any version specifiers included with requirements strings
    for operator in ' <>=!':
        project_name = project_name.split(operator)[0]

    return project_name
