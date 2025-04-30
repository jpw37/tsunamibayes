"""Miscellaneous functions for the mudpy engine."""

# Copyright (c) 2004-2021 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

import codecs
import datetime
import os
import random
import re
import signal
import socket
import sys
import syslog
import time
import traceback
import unicodedata

import mudpy


class Element:

    """An element of the universe."""

    def __init__(self, key, universe, origin=None):
        """Set up a new element."""

        # keep track of our key name
        self.key = key

        # keep track of what universe it's loading into
        self.universe = universe

        # set of facet keys from the universe
        self.facethash = dict()

        # not owned by a user by default (used for avatars)
        self.owner = None

        # no contents in here by default
        self.contents = {}

        if self.key.find(".") > 0:
            self.group, self.subkey = self.key.split(".")[-2:]
        else:
            self.group = "other"
            self.subkey = self.key
        if self.group not in self.universe.groups:
            self.universe.groups[self.group] = {}

        # get an appropriate origin
        if not origin:
            self.universe.add_group(self.group)
            origin = self.universe.files[
                    self.universe.origins[self.group]["fallback"]]

        # record or reset a pointer to the origin file
        self.origin = self.universe.files[origin.source]

        # add or replace this element in the universe
        self.universe.contents[self.key] = self
        self.universe.groups[self.group][self.subkey] = self

    def reload(self):
        """Create a new element and replace this one."""
        args = (self.key, self.universe, self.origin)
        self.destroy()
        Element(*args)

    def destroy(self):
        """Remove an element from the universe and destroy it."""
        for facet in dict(self.facethash):
            self.remove_facet(facet)
        del self.universe.groups[self.group][self.subkey]
        del self.universe.contents[self.key]
        del self

    def facets(self):
        """Return a list of non-inherited facets for this element."""
        return self.facethash

    def has_facet(self, facet):
        """Return whether the non-inherited facet exists."""
        return facet in self.facets()

    def remove_facet(self, facet):
        """Remove a facet from the element."""
        if ".".join((self.key, facet)) in self.origin.data:
            del self.origin.data[".".join((self.key, facet))]
        if facet in self.facethash:
            del self.facethash[facet]
        self.origin.modified = True

    def ancestry(self):
        """Return a list of the element's inheritance lineage."""
        if self.has_facet("inherit"):
            ancestry = self.get("inherit")
            if not ancestry:
                ancestry = []
            for parent in ancestry[:]:
                ancestors = self.universe.contents[parent].ancestry()
                for ancestor in ancestors:
                    if ancestor not in ancestry:
                        ancestry.append(ancestor)
            return ancestry
        else:
            return []

    def get(self, facet, default=None):
        """Retrieve values."""
        if default is None:
            default = ""
        try:
            return self.origin.data[".".join((self.key, facet))]
        except (KeyError, TypeError):
            pass
        if self.has_facet("inherit"):
            for ancestor in self.ancestry():
                if self.universe.contents[ancestor].has_facet(facet):
                    return self.universe.contents[ancestor].get(facet)
        else:
            return default

    def set(self, facet, value):
        """Set values."""
        if not self.origin.is_writeable() and not self.universe.loading:
            # break if there is an attempt to update an element from a
            # read-only file, unless the universe is in the midst of loading
            # updated data from files
            raise PermissionError("Altering elements in read-only files is "
                                  "disallowed")
        # Coerce some values to appropriate data types
        # TODO(fungi) Move these to a separate validation mechanism
        if facet in ["loglevel"]:
            value = int(value)
        elif facet in ["administrator"]:
            value = bool(value)

        # The canonical node for this facet within its origin
        node = ".".join((self.key, facet))

        if node not in self.origin.data or self.origin.data[node] != value:
            # Be careful to only update the origin's contents when required,
            # since that affects whether the backing file gets written
            self.origin.data[node] = value
            self.origin.modified = True

        # Make sure this facet is included in the element's facets
        self.facethash[facet] = self.origin.data[node]

    def append(self, facet, value):
        """Append value to a list."""
        newlist = self.get(facet)
        if not newlist:
            newlist = []
        if type(newlist) is not list:
            newlist = list(newlist)
        newlist.append(value)
        self.set(facet, newlist)

    def send(
        self,
        message,
        eol="$(eol)",
        raw=False,
        flush=False,
        add_prompt=True,
        just_prompt=False,
        add_terminator=False,
        prepend_padding=True
    ):
        """Convenience method to pass messages to an owner."""
        if self.owner:
            self.owner.send(
                message,
                eol,
                raw,
                flush,
                add_prompt,
                just_prompt,
                add_terminator,
                prepend_padding
            )

    def is_restricted(self):
        """Boolean check whether command is administrative or debugging."""
        return bool(self.get("administrative") or self.get("debugging"))

    def is_admin(self):
        """Boolean check whether an actor is controlled by an admin owner."""
        return self.owner and self.owner.is_admin()

    def can_run(self, command):
        """Check if the user can run this command object."""

        # has to be in the commands group
        if command not in self.universe.groups["command"].values():
            return False

        # debugging commands are not allowed outside debug mode
        if command.get("debugging") and not self.universe.debug_mode():
            return False

        # avatars of administrators can run any command
        if self.is_admin():
            return True

        # everyone can run non-administrative commands
        if not command.is_restricted():
            return True

        # otherwise the command cannot be run by this actor
        return False

    def update_location(self):
        """Make sure the location's contents contain this element."""
        area = self.get("location")
        if area in self.universe.contents:
            self.universe.contents[area].contents[self.key] = self

    def clean_contents(self):
        """Make sure the element's contents aren't bogus."""
        for element in self.contents.values():
            if element.get("location") != self.key:
                del self.contents[element.key]

    def go_to(self, area):
        """Relocate the element to a specific area."""
        current = self.get("location")
        if current and self.key in self.universe.contents[current].contents:
            del universe.contents[current].contents[self.key]
        if area in self.universe.contents:
            self.set("location", area)
        self.universe.contents[area].contents[self.key] = self
        self.look_at(area)

    def go_home(self):
        """Relocate the element to its default location."""
        self.go_to(self.get("default_location"))
        self.echo_to_location(
            "You suddenly realize that " + self.get("name") + " is here."
        )

    def move_direction(self, direction):
        """Relocate the element in a specified direction."""
        motion = self.universe.contents["mudpy.movement.%s" % direction]
        enter_term = motion.get("enter_term")
        exit_term = motion.get("exit_term")
        self.echo_to_location("%s exits %s." % (self.get("name"), exit_term))
        self.send("You exit %s." % exit_term, add_prompt=False)
        self.go_to(
            self.universe.contents[
                self.get("location")].link_neighbor(direction)
        )
        self.echo_to_location("%s arrives from %s." % (
            self.get("name"), enter_term))

    def look_at(self, key):
        """Show an element to another element."""
        if self.owner:
            element = self.universe.contents[key]
            message = ""
            name = element.get("name")
            if name:
                message += "$(cyn)" + name + "$(nrm)$(eol)"
            description = element.get("description")
            if description:
                message += description + "$(eol)"
            portal_list = list(element.portals().keys())
            if portal_list:
                portal_list.sort()
                message += "$(cyn)[ Exits: " + ", ".join(
                    portal_list
                ) + " ]$(nrm)$(eol)"
            for element in self.universe.contents[
                self.get("location")
            ].contents.values():
                if element.get("is_actor") and element is not self:
                    message += "$(yel)" + element.get(
                        "name"
                    ) + " is here.$(nrm)$(eol)"
                elif element is not self:
                    message += "$(grn)" + element.get(
                        "impression"
                    ) + "$(nrm)$(eol)"
            self.send(message)

    def portals(self):
        """Map the portal directions for an area to neighbors."""
        portals = {}
        if re.match(r"""^area\.-?\d+,-?\d+,-?\d+$""", self.key):
            coordinates = [(int(x))
                           for x in self.key.split(".")[-1].split(",")]
            offsets = dict(
                (x,
                 self.universe.contents["mudpy.movement.%s" % x].get("vector")
                 ) for x in self.universe.directions)
            for portal in self.get("gridlinks"):
                adjacent = map(lambda c, o: c + o,
                               coordinates, offsets[portal])
                neighbor = "area." + ",".join(
                    [(str(x)) for x in adjacent]
                )
                if neighbor in self.universe.contents:
                    portals[portal] = neighbor
        for facet in self.facets():
            if facet.startswith("link_"):
                neighbor = self.get(facet)
                if neighbor in self.universe.contents:
                    portal = facet.split("_")[1]
                    portals[portal] = neighbor
        return portals

    def link_neighbor(self, direction):
        """Return the element linked in a given direction."""
        portals = self.portals()
        if direction in portals:
            return portals[direction]

    def echo_to_location(self, message):
        """Show a message to other elements in the current location."""
        for element in self.universe.contents[
            self.get("location")
        ].contents.values():
            if element is not self:
                element.send(message)


class Universe:

    """The universe."""

    def __init__(self, filename="", load=False):
        """Initialize the universe."""
        self.groups = {}
        self.contents = {}
        self.directions = set()
        self.loading = False
        self.loglines = []
        self.origins = {}
        self.reload_flag = False
        self.setup_loglines = []
        self.startdir = os.getcwd()
        self.terminate_flag = False
        self.userlist = []
        self.versions = None
        if not filename:
            possible_filenames = [
                "etc/mudpy.yaml",
                "/usr/local/mudpy/etc/mudpy.yaml",
                "/usr/local/etc/mudpy.yaml",
                "/etc/mudpy/mudpy.yaml",
                "/etc/mudpy.yaml"
            ]
            for filename in possible_filenames:
                if os.access(filename, os.R_OK):
                    break
        if not os.path.isabs(filename):
            filename = os.path.join(self.startdir, filename)
        self.filename = filename
        if load:
            # make sure to preserve any accumulated log entries during load
            self.setup_loglines += self.load()

    def load(self):
        """Load universe data from persistent storage."""

        # while loading, it's safe to update elements from read-only files
        self.loading = True

        # it's possible for this to enter before logging configuration is read
        pending_loglines = []

        # start populating the (re)files dict from the base config
        self.files = {}
        mudpy.data.Data(self.filename, self)

        # load default storage locations for groups
        if hasattr(self, "contents") and "mudpy.filing" in self.contents:
            self.origins.update(self.contents["mudpy.filing"].get(
                "groups", {}))

        # add some builtin groups we know we'll need
        for group in ("account", "actor", "internal"):
            self.add_group(group)

        # make a list of inactive avatars
        inactive_avatars = []
        for account in self.groups.get("account", {}).values():
            for avatar in account.get("avatars"):
                try:
                    inactive_avatars.append(self.contents[avatar])
                except KeyError:
                    pending_loglines.append((
                        'Missing avatar "%s", possible data corruption' %
                        avatar, 6))
        for user in self.userlist:
            if user.avatar in inactive_avatars:
                inactive_avatars.remove(user.avatar)

        # another pass to straighten out all the element contents
        for element in self.contents.values():
            element.update_location()
            element.clean_contents()

        # warn when debug mode has been engaged
        if self.debug_mode():
            pending_loglines.append((
                "WARNING: Unsafe debugging mode is enabled!", 6))

        # done loading, so disallow updating elements from read-only files
        self.loading = False

        return pending_loglines

    def new(self):
        """Create a new, empty Universe (the Big Bang)."""
        new_universe = Universe()
        for attribute in vars(self).keys():
            setattr(new_universe, attribute, getattr(self, attribute))
        new_universe.reload_flag = False
        del self
        return new_universe

    def save(self):
        """Save the universe to persistent storage."""
        for key in self.files:
            self.files[key].save()

    def initialize_server_socket(self):
        """Create and open the listening socket."""

        # need to know the local address and port number for the listener
        host = self.contents["mudpy.network"].get("host")
        port = self.contents["mudpy.network"].get("port")

        # if no host was specified, bind to the loopback address (preferring
        # ipv6)
        if not host:
            if socket.has_ipv6:
                host = "::1"
            else:
                host = "127.0.0.1"

        # figure out if this is ipv4 or v6
        family = socket.getaddrinfo(host, port)[0][0]
        if family is socket.AF_INET6 and not socket.has_ipv6:
            log("No support for IPv6 address %s (use IPv4 instead)." % host)

        # create a new stream-type socket object
        self.listening_socket = socket.socket(family, socket.SOCK_STREAM)

        # set the socket options to allow existing open ones to be
        # reused (fixes a bug where the server can't bind for a minute
        # when restarting on linux systems)
        self.listening_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )

        # bind the socket to to our desired server ipa and port
        self.listening_socket.bind((host, port))

        # disable blocking so we can proceed whether or not we can
        # send/receive
        self.listening_socket.setblocking(0)

        # start listening on the socket
        self.listening_socket.listen(1)

        # note that we're now ready for user connections
        log("Listening for Telnet connections on %s port %s" % (
                host, str(port)))

    def get_time(self):
        """Convenience method to get the elapsed time counter."""
        try:
            return self.groups["internal"]["counters"].get("elapsed", 0)
        except KeyError:
            return 0

    def set_time(self, elapsed):
        """Convenience method to set the elapsed time counter."""
        try:
            self.groups["internal"]["counters"].set("elapsed", elapsed)
        except KeyError:
            # add an element for counters if it doesn't exist
            Element("internal.counters", universe)
            self.groups["internal"]["counters"].set("elapsed", elapsed)

    def add_group(self, group, fallback=None):
        """Set up group tracking/metadata."""
        if group not in self.origins:
            self.origins[group] = {}
        if not fallback:
            fallback = mudpy.data.find_file(
                    ".".join((group, "yaml")), universe=self)
        if "fallback" not in self.origins[group]:
            self.origins[group]["fallback"] = fallback
        flags = self.origins[group].get("flags", None)
        if fallback not in self.files:
            mudpy.data.Data(fallback, self, flags=flags)

    def debug_mode(self):
        """Boolean method to indicate whether unsafe debugging is enabled."""
        return self.groups["mudpy"]["limit"].get("debug", False)


class User:

    """This is a connected user."""

    def __init__(self):
        """Default values for the in-memory user variables."""
        self.account = None
        self.address = ""
        self.authenticated = False
        self.avatar = None
        self.choice = ""
        self.columns = 79
        self.connection = None
        self.error = ""
        self.input_queue = []
        self.last_address = ""
        self.last_input = universe.get_time()
        self.menu_choices = {}
        self.menu_seen = False
        self.negotiation_pause = 0
        self.output_queue = []
        self.partial_input = b""
        self.password_tries = 0
        self.rows = 23
        self.state = "telopt_negotiation"
        self.telopts = {}
        self.ttype = None
        self.universe = universe

    def quit(self):
        """Log, close the connection and remove."""
        if self.account:
            name = self.account.get("name", self)
        else:
            name = self
        log("Logging out %s" % name, 2)
        self.deactivate_avatar()
        self.connection.close()
        self.remove()

    def check_idle(self):
        """Warn or disconnect idle users as appropriate."""
        idletime = universe.get_time() - self.last_input
        linkdead_dict = universe.contents[
            "mudpy.timing.idle.disconnect"].facets()
        if self.state in linkdead_dict:
            linkdead_state = self.state
        else:
            linkdead_state = "default"
        if idletime > linkdead_dict[linkdead_state]:
            self.send(
                "$(eol)$(red)You've done nothing for far too long... goodbye!"
                + "$(nrm)$(eol)",
                flush=True,
                add_prompt=False
            )
            logline = "Disconnecting "
            if self.account and self.account.get("name"):
                logline += self.account.get("name")
            else:
                logline += "an unknown user"
            logline += (" after idling too long in the " + self.state
                        + " state.")
            log(logline, 2)
            self.state = "disconnecting"
            self.menu_seen = False
        idle_dict = universe.contents["mudpy.timing.idle.warn"].facets()
        if self.state in idle_dict:
            idle_state = self.state
        else:
            idle_state = "default"
        if idletime == idle_dict[idle_state]:
            self.send(
                "$(eol)$(red)If you continue to be unproductive, "
                + "you'll be shown the door...$(nrm)$(eol)"
            )

    def reload(self):
        """Save, load a new user and relocate the connection."""

        # copy old attributes
        attributes = self.__dict__

        # get out of the list
        self.remove()

        # get rid of the old user object
        del self

        # create a new user object
        new_user = User()

        # set everything equivalent
        new_user.__dict__ = attributes

        # the avatar needs a new owner
        if new_user.avatar:
            new_user.account = universe.contents[new_user.account.key]
            new_user.avatar = universe.contents[new_user.avatar.key]
            new_user.avatar.owner = new_user

        # add it to the list
        universe.userlist.append(new_user)

    def replace_old_connections(self):
        """Disconnect active users with the same name."""

        # the default return value
        return_value = False

        # iterate over each user in the list
        for old_user in universe.userlist:

            # the name is the same but it's not us
            if hasattr(
               old_user, "account"
               ) and old_user.account and old_user.account.get(
                "name"
            ) == self.account.get(
                "name"
            ) and old_user is not self:

                # make a note of it
                log(
                    "User " + self.account.get(
                        "name"
                    ) + " reconnected--closing old connection to "
                    + old_user.address + ".",
                    2
                )
                old_user.send(
                    "$(eol)$(red)New connection from " + self.address
                    + ". Terminating old connection...$(nrm)$(eol)",
                    flush=True,
                    add_prompt=False
                )

                # close the old connection
                old_user.connection.close()

                # replace the old connection with this one
                old_user.send(
                    "$(eol)$(red)Taking over old connection from "
                    + old_user.address + ".$(nrm)"
                )
                old_user.connection = self.connection
                old_user.last_address = old_user.address
                old_user.address = self.address
                old_user.telopts = self.telopts
                old_user.adjust_echoing()

                # take this one out of the list and delete
                self.remove()
                del self
                return_value = True
                break

        # true if an old connection was replaced, false if not
        return return_value

    def authenticate(self):
        """Flag the user as authenticated and disconnect duplicates."""
        if self.state != "authenticated":
            self.authenticated = True
            log("User %s authenticated for account %s." % (
                    self, self.account.subkey), 2)
            if ("mudpy.limit" in universe.contents and self.account.subkey in
                    universe.contents["mudpy.limit"].get("admins")):
                self.account.set("administrator", True)
                log("Account %s is an administrator." % (
                        self.account.subkey), 2)

    def show_menu(self):
        """Send the user their current menu."""
        if not self.menu_seen:
            self.menu_choices = get_menu_choices(self)
            self.send(
                get_menu(self.state, self.error, self.menu_choices),
                "",
                add_terminator=True
            )
            self.menu_seen = True
            self.error = False
            self.adjust_echoing()

    def prompt(self):
        """"Generate and return an input prompt."""

        # Start with the user's preference, if one was provided
        prompt = self.account.get("prompt")

        # If the user has not set a prompt, then immediately return the default
        # provided for the current state
        if not prompt:
            return get_menu_prompt(self.state)

        # Allow including the World clock state
        if "$_(time)" in prompt:
            prompt = prompt.replace(
                "$_(time)",
                str(universe.get_time()))

        # Append a single space for clear separation from user input
        if prompt[-1] != " ":
            prompt = "%s " % prompt

        # Return the cooked prompt
        return prompt

    def adjust_echoing(self):
        """Adjust echoing to match state menu requirements."""
        if mudpy.telnet.is_enabled(self, mudpy.telnet.TELOPT_ECHO,
                                   mudpy.telnet.US):
            if menu_echo_on(self.state):
                mudpy.telnet.disable(self, mudpy.telnet.TELOPT_ECHO,
                                     mudpy.telnet.US)
        elif not menu_echo_on(self.state):
            mudpy.telnet.enable(self, mudpy.telnet.TELOPT_ECHO,
                                mudpy.telnet.US)

    def remove(self):
        """Remove a user from the list of connected users."""
        log("Disconnecting account %s." % self, 0)
        universe.userlist.remove(self)

    def send(
        self,
        output,
        eol="$(eol)",
        raw=False,
        flush=False,
        add_prompt=True,
        just_prompt=False,
        add_terminator=False,
        prepend_padding=True
    ):
        """Send arbitrary text to a connected user."""

        # unless raw mode is on, clean it up all nice and pretty
        if not raw:

            # strip extra $(eol) off if present
            while output.startswith("$(eol)"):
                output = output[6:]
            while output.endswith("$(eol)"):
                output = output[:-6]
            extra_lines = output.find("$(eol)$(eol)$(eol)")
            while extra_lines > -1:
                output = output[:extra_lines] + output[extra_lines + 6:]
                extra_lines = output.find("$(eol)$(eol)$(eol)")

            # start with a newline, append the message, then end
            # with the optional eol string passed to this function
            # and the ansi escape to return to normal text
            if not just_prompt and prepend_padding:
                if (not self.output_queue or not
                        self.output_queue[-1].endswith(b"\r\n")):
                    output = "$(eol)" + output
                elif not self.output_queue[-1].endswith(
                    b"\r\n\x1b[0m\r\n"
                ) and not self.output_queue[-1].endswith(
                    b"\r\n\r\n"
                ):
                    output = "$(eol)" + output
            output += eol + chr(27) + "[0m"

            # tack on a prompt if active
            if self.state == "active":
                if not just_prompt:
                    output += "$(eol)"
                if add_prompt:
                    output += self.prompt()
                    mode = self.avatar.get("mode")
                    if mode:
                        output += "(" + mode + ") "

            # find and replace macros in the output
            output = replace_macros(self, output)

            # wrap the text at the client's width (min 40, 0 disables)
            if self.columns:
                if self.columns < 40:
                    wrap = 40
                else:
                    wrap = self.columns
                output = wrap_ansi_text(output, wrap)

            # if supported by the client, encode it utf-8
            if mudpy.telnet.is_enabled(self, mudpy.telnet.TELOPT_BINARY,
                                       mudpy.telnet.US):
                encoded_output = output.encode("utf-8")

            # otherwise just send ascii
            else:
                encoded_output = output.encode("ascii", "replace")

            # end with a terminator if requested
            if add_prompt or add_terminator:
                if mudpy.telnet.is_enabled(
                        self, mudpy.telnet.TELOPT_EOR, mudpy.telnet.US):
                    encoded_output += mudpy.telnet.telnet_proto(
                        mudpy.telnet.IAC, mudpy.telnet.EOR)
                elif not mudpy.telnet.is_enabled(
                        self, mudpy.telnet.TELOPT_SGA, mudpy.telnet.US):
                    encoded_output += mudpy.telnet.telnet_proto(
                        mudpy.telnet.IAC, mudpy.telnet.GA)

            # and tack it onto the queue
            self.output_queue.append(encoded_output)

            # if this is urgent, flush all pending output
            if flush:
                self.flush()

        # just dump raw bytes as requested
        else:
            self.output_queue.append(output)
            self.flush()

    def pulse(self):
        """All the things to do to the user per increment."""

        # if the world is terminating, disconnect
        if universe.terminate_flag:
            self.state = "disconnecting"
            self.menu_seen = False

        # check for an idle connection and act appropriately
        else:
            self.check_idle()

        # ask the client for their current terminal type (RFC 1091); it's None
        # if it's not been initialized, the empty string if it has but the
        # output was indeterminate, "UNKNOWN" if the client specified it has no
        # terminal types to supply
        if self.ttype is None:
            mudpy.telnet.request_ttype(self)

        # if output is paused, decrement the counter
        if self.state == "telopt_negotiation":
            if self.negotiation_pause:
                self.negotiation_pause -= 1
            else:
                self.state = "entering_account_name"

        # show the user a menu as needed
        elif not self.state == "active":
            self.show_menu()

        # flush any pending output in the queue
        self.flush()

        # disconnect users with the appropriate state
        if self.state == "disconnecting":
            self.quit()

        # check for input and add it to the queue
        self.enqueue_input()

        # there is input waiting in the queue
        if self.input_queue:
            handle_user_input(self)

    def flush(self):
        """Try to send the last item in the queue and remove it."""
        if self.output_queue:
            try:
                self.connection.send(self.output_queue[0])
            except (BrokenPipeError, ConnectionResetError):
                if self.account and self.account.get("name"):
                    account = self.account.get("name")
                else:
                    account = "an unknown user"
                self.state = "disconnecting"
                log("Disconnected while sending to %s." % account, 7)
            del self.output_queue[0]

    def enqueue_input(self):
        """Process and enqueue any new input."""

        # check for some input
        try:
            raw_input = self.connection.recv(1024)
        except OSError:
            raw_input = b""

        # we got something
        if raw_input:

            # tack this on to any previous partial
            self.partial_input += raw_input

            # reply to and remove any IAC negotiation codes
            mudpy.telnet.negotiate_telnet_options(self)

            # separate multiple input lines
            new_input_lines = self.partial_input.split(b"\r\0")
            if len(new_input_lines) == 1:
                new_input_lines = new_input_lines[0].split(b"\r\n")

            # if input doesn't end in a newline, replace the
            # held partial input with the last line of it
            if not (
                    self.partial_input.endswith(b"\r\0") or
                    self.partial_input.endswith(b"\r\n")):
                self.partial_input = new_input_lines.pop()

            # otherwise, chop off the extra null input and reset
            # the held partial input
            else:
                new_input_lines.pop()
                self.partial_input = b""

            # iterate over the remaining lines
            for line in new_input_lines:

                # strip off extra whitespace
                line = line.strip()

                # log non-printable characters remaining
                if not mudpy.telnet.is_enabled(
                        self, mudpy.telnet.TELOPT_BINARY, mudpy.telnet.HIM):
                    asciiline = bytes([x for x in line if 32 <= x <= 126])
                    if line != asciiline:
                        logline = "Non-ASCII characters from "
                        if self.account and self.account.get("name"):
                            logline += self.account.get("name") + ": "
                        else:
                            logline += "unknown user: "
                        logline += repr(line)
                        log(logline, 4)
                        line = asciiline

                try:
                    line = line.decode("utf-8")
                except UnicodeDecodeError:
                    logline = "Non-UTF-8 sequence from "
                    if self.account and self.account.get("name"):
                        logline += self.account.get("name") + ": "
                    else:
                        logline += "unknown user: "
                    logline += repr(line)
                    log(logline, 4)
                    return

                line = unicodedata.normalize("NFKC", line)

                # put on the end of the queue
                self.input_queue.append(line)

    def new_avatar(self):
        """Instantiate a new, unconfigured avatar for this user."""
        counter = 0
        while ("avatar_%s_%s" % (self.account.get("name"), counter)
                in universe.groups.get("actor", {}).keys()):
            counter += 1
        self.avatar = Element(
            "actor.avatar_%s_%s" % (self.account.get("name"), counter),
            universe)
        self.avatar.append("inherit", "archetype.avatar")
        self.account.append("avatars", self.avatar.key)
        log("Created new avatar %s for user %s." % (
                self.avatar.key, self.account.get("name")), 0)

    def delete_avatar(self, avatar):
        """Remove an avatar from the world and from the user's list."""
        if self.avatar is universe.contents[avatar]:
            self.avatar = None
        log("Deleting avatar %s for user %s." % (
                avatar, self.account.get("name")), 0)
        universe.contents[avatar].destroy()
        avatars = self.account.get("avatars")
        avatars.remove(avatar)
        self.account.set("avatars", avatars)

    def activate_avatar_by_index(self, index):
        """Enter the world with a particular indexed avatar."""
        self.avatar = universe.contents[
            self.account.get("avatars")[index]]
        self.avatar.owner = self
        self.state = "active"
        log("Activated avatar %s (%s)." % (
                self.avatar.get("name"), self.avatar.key), 0)
        self.avatar.go_home()

    def deactivate_avatar(self):
        """Have the active avatar leave the world."""
        if self.avatar:
            log("Deactivating avatar %s (%s) for user %s." % (
                    self.avatar.get("name"), self.avatar.key,
                    self.account.get("name")), 0)
            current = self.avatar.get("location")
            if current:
                self.avatar.set("default_location", current)
                self.avatar.echo_to_location(
                    "You suddenly wonder where " + self.avatar.get(
                        "name"
                    ) + " went."
                )
                del universe.contents[current].contents[self.avatar.key]
                self.avatar.remove_facet("location")
            self.avatar.owner = None
            self.avatar = None

    def destroy(self):
        """Destroy the user and associated avatars."""
        for avatar in self.account.get("avatars"):
            self.delete_avatar(avatar)
        log("Destroying account %s for user %s." % (
                self.account.get("name"), self), 0)
        self.account.destroy()

    def list_avatar_names(self):
        """List names of assigned avatars."""
        avatars = []
        for avatar in self.account.get("avatars"):
            try:
                avatars.append(universe.contents[avatar].get("name"))
            except KeyError:
                log('Missing avatar "%s", possible data corruption.' %
                    avatar, 6)
        return avatars

    def is_admin(self):
        """Boolean check whether user's account is an admin."""
        return self.account.get("administrator", False)


def broadcast(message, add_prompt=True):
    """Send a message to all connected users."""
    for each_user in universe.userlist:
        each_user.send("$(eol)" + message, add_prompt=add_prompt)


def log(message, level=0):
    """Log a message."""

    # a couple references we need
    if "mudpy.log" in universe.contents:
        file_name = universe.contents["mudpy.log"].get("file", "")
        max_log_lines = universe.contents["mudpy.log"].get("lines", 0)
        syslog_name = universe.contents["mudpy.log"].get("syslog", "")
    else:
        file_name = ""
        max_log_lines = 0
        syslog_name = ""
    timestamp = datetime.datetime.now().isoformat(' ')

    # turn the message into a list of nonempty lines
    lines = [x for x in [(x.rstrip()) for x in message.split("\n")] if x != ""]

    # send the timestamp and line to a file
    if file_name:
        if not os.path.isabs(file_name):
            file_name = os.path.join(universe.startdir, file_name)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        file_descriptor = codecs.open(file_name, "a", "utf-8")
        for line in lines:
            file_descriptor.write(timestamp + " " + line + "\n")
        file_descriptor.flush()
        file_descriptor.close()

    # send the timestamp and line to standard output
    if ("mudpy.log" in universe.contents and
            universe.contents["mudpy.log"].get("stdout")):
        for line in lines:
            print(timestamp + " " + line)

    # send the line to the system log
    if syslog_name:
        syslog.openlog(
            syslog_name.encode("utf-8"),
            syslog.LOG_PID,
            syslog.LOG_INFO | syslog.LOG_DAEMON
        )
        for line in lines:
            syslog.syslog(line)
        syslog.closelog()

    # display to connected administrators
    for user in universe.userlist:
        if (
                user.state == "active"
                and user.is_admin()
                and user.account.get("loglevel", 0) <= level):
            # iterate over every line in the message
            full_message = ""
            for line in lines:
                full_message += (
                    "$(bld)$(red)" + timestamp + " "
                    + line.replace("$(", "$_(") + "$(nrm)$(eol)")
            user.send(full_message, flush=True)

    # add to the recent log list
    for line in lines:
        while 0 < len(universe.loglines) >= max_log_lines:
            del universe.loglines[0]
        universe.loglines.append((timestamp + " " + line, level))


def get_loglines(level, start, stop):
    """Return a specific range of loglines filtered by level."""

    # filter the log lines
    loglines = [x for x in universe.loglines if x[1] >= level]

    # we need these in several places
    total_count = str(len(universe.loglines))
    filtered_count = len(loglines)

    # don't proceed if there are no lines
    if filtered_count:

        # can't start before the beginning or at the end
        if start > filtered_count:
            start = filtered_count
        if start < 1:
            start = 1

        # can't stop before we start
        if stop > start:
            stop = start
        elif stop < 1:
            stop = 1

        # some preamble
        message = (
            "There are %s log lines in memory and %s at or above level %s. "
            "The matching lines from %s to %s are:$(eol)$(eol)" %
            (total_count, filtered_count, level, stop, start))

        # add the text from the selected lines
        if stop > 1:
            range_lines = loglines[-start:-(stop - 1)]
        else:
            range_lines = loglines[-start:]
        for line in range_lines:
            message += "   (%s) %s$(eol)" % (
                line[1], line[0].replace("$(", "$_("))

    # there were no lines
    else:
        message = "None of the %s lines in memory matches your request." % (
            total_count)

    # pass it back
    return message


def glyph_columns(character):
    """Convenience function to return the column width of a glyph."""
    if unicodedata.east_asian_width(character) in "FW":
        return 2
    else:
        return 1


def wrap_ansi_text(text, width):
    """Wrap text with arbitrary width while ignoring ANSI colors."""

    # the current position in the entire text string, including all
    # characters, printable or otherwise
    abs_pos = 0

    # the current text position relative to the beginning of the line,
    # ignoring color escape sequences
    rel_pos = 0

    # the absolute and relative positions of the most recent whitespace
    # character
    last_abs_whitespace = 0
    last_rel_whitespace = 0

    # whether the current character is part of a color escape sequence
    escape = False

    # normalize any potentially composited unicode before we count it
    text = unicodedata.normalize("NFKC", text)

    # iterate over each character from the beginning of the text
    for each_character in text:

        # the current character is the escape character
        if each_character == "\x1b" and not escape:
            escape = True
            rel_pos -= 1

        # the current character is within an escape sequence
        elif escape:
            rel_pos -= 1
            if each_character == "m":
                # the current character is m, which terminates the
                # escape sequence
                escape = False

        # the current character is a space
        elif each_character == " ":
            last_abs_whitespace = abs_pos
            last_rel_whitespace = rel_pos

        # the current character is a newline, so reset the relative
        # position too (start a new line)
        elif each_character == "\n":
            rel_pos = 0
            last_abs_whitespace = abs_pos
            last_rel_whitespace = rel_pos

        # the current character meets the requested maximum line width, so we
        # need to wrap unless the current word is wider than the terminal (in
        # which case we let it do the wrapping instead)
        if last_rel_whitespace != 0 and (rel_pos > width or (
                rel_pos > width - 1 and glyph_columns(each_character) == 2)):

            # insert an eol in place of the last space
            text = (text[:last_abs_whitespace] + "\r\n" +
                    text[last_abs_whitespace + 1:])

            # increase the absolute position because an eol is two
            # characters but the space it replaced was only one
            abs_pos += 1

            # now we're at the beginning of a new line, plus the
            # number of characters wrapped from the previous line
            rel_pos -= last_rel_whitespace
            last_rel_whitespace = 0

        # as long as the character is not a carriage return and the
        # other above conditions haven't been met, count it as a
        # printable character
        elif each_character != "\r":
            rel_pos += glyph_columns(each_character)
            if each_character in (" ", "\n"):
                last_abs_whitespace = abs_pos
                last_rel_whitespace = rel_pos

        # increase the absolute position for every character
        abs_pos += 1

    # return the newly-wrapped text
    return text


def weighted_choice(data):
    """Takes a dict weighted by value and returns a random key."""

    # this will hold our expanded list of keys from the data
    expanded = []

    # create the expanded list of keys
    for key in data.keys():
        for _count in range(data[key]):
            expanded.append(key)

    # return one at random
    # Allow the random.randrange() call in bandit since it's not used for
    # security/cryptographic purposes
    return random.choice(expanded)  # nosec


def random_name():
    """Returns a random character name."""

    # the vowels and consonants needed to create romaji syllables
    vowels = [
        "a",
        "i",
        "u",
        "e",
        "o"
    ]
    consonants = [
        "'",
        "k",
        "z",
        "s",
        "sh",
        "z",
        "j",
        "t",
        "ch",
        "ts",
        "d",
        "n",
        "h",
        "f",
        "m",
        "y",
        "r",
        "w"
    ]

    # this dict will hold our weighted list of syllables
    syllables = {}

    # generate the list with an even weighting
    for consonant in consonants:
        for vowel in vowels:
            syllables[consonant + vowel] = 1

    # we'll build the name into this string
    name = ""

    # create a name of random length from the syllables
    # Allow the random.randrange() call in bandit since it's not used for
    # security/cryptographic purposes
    for _syllable in range(random.randrange(2, 6)):  # nosec
        name += weighted_choice(syllables)

    # strip any leading quotemark, capitalize and return the name
    return name.strip("'").capitalize()


def replace_macros(user, text, is_input=False):
    """Replaces macros in text output."""

    # third person pronouns
    pronouns = {
        "female": {"obj": "her", "pos": "hers", "sub": "she"},
        "male": {"obj": "him", "pos": "his", "sub": "he"},
        "neuter": {"obj": "it", "pos": "its", "sub": "it"}
    }

    # a dict of replacement macros
    macros = {
        "eol": "\r\n",
        "bld": chr(27) + "[1m",
        "nrm": chr(27) + "[0m",
        "blk": chr(27) + "[30m",
        "blu": chr(27) + "[34m",
        "cyn": chr(27) + "[36m",
        "grn": chr(27) + "[32m",
        "mgt": chr(27) + "[35m",
        "red": chr(27) + "[31m",
        "yel": chr(27) + "[33m",
    }

    # add dynamic macros where possible
    if user.account:
        account_name = user.account.get("name")
        if account_name:
            macros["account"] = account_name
    if user.avatar:
        avatar_gender = user.avatar.get("gender")
        if avatar_gender:
            macros["tpop"] = pronouns[avatar_gender]["obj"]
            macros["tppp"] = pronouns[avatar_gender]["pos"]
            macros["tpsp"] = pronouns[avatar_gender]["sub"]

    # loop until broken
    while True:

        # find and replace per the macros dict
        macro_start = text.find("$(")
        if macro_start == -1:
            break
        macro_end = text.find(")", macro_start) + 1
        macro = text[macro_start + 2:macro_end - 1]
        if macro in macros.keys():
            replacement = macros[macro]

        # this is how we handle local file inclusion (dangerous!)
        elif macro.startswith("inc:"):
            incfile = mudpy.data.find_file(macro[4:], universe=universe)
            if os.path.exists(incfile):
                replacement = ""
                with codecs.open(incfile, "r", "utf-8") as incfd:
                    for line in incfd:
                        if line.endswith("\n") and not line.endswith("\r\n"):
                            line = line.replace("\n", "\r\n")
                        replacement += line
                    # lose the trailing eol
                    replacement = replacement[:-2]
            else:
                replacement = ""
                log("Couldn't read included " + incfile + " file.", 7)

        # if we get here, log and replace it with null
        else:
            replacement = ""
            if not is_input:
                log("Unexpected replacement macro " +
                    macro + " encountered.", 6)

        # and now we act on the replacement
        text = text.replace("$(" + macro + ")", replacement)

    # replace the look-like-a-macro sequence
    text = text.replace("$_(", "$(")

    return text


def escape_macros(value):
    """Escapes replacement macros in text."""
    if type(value) is str:
        return value.replace("$(", "$_(")
    else:
        return value


def first_word(text, separator=" "):
    """Returns a tuple of the first word and the rest."""
    if text:
        if text.find(separator) > 0:
            return text.split(separator, 1)
        else:
            return text, ""
    else:
        return "", ""


def on_pulse():
    """The things which should happen on each pulse, aside from reloads."""

    # increase the elapsed increment counter
    universe.set_time(universe.get_time() + 1)

    # update the log every now and then
    if not universe.groups["internal"]["counters"].get("mark"):
        log(str(len(universe.userlist)) + " connection(s)")
        universe.groups["internal"]["counters"].set(
            "mark", universe.contents["mudpy.timing"].get("status")
        )
    else:
        universe.groups["internal"]["counters"].set(
            "mark", universe.groups["internal"]["counters"].get(
                "mark"
            ) - 1
        )

    # periodically save everything
    if not universe.groups["internal"]["counters"].get("save"):
        universe.save()
        universe.groups["internal"]["counters"].set(
            "save", universe.contents["mudpy.timing"].get("save")
        )
    else:
        universe.groups["internal"]["counters"].set(
            "save", universe.groups["internal"]["counters"].get(
                "save"
            ) - 1
        )

    # open the listening socket if it hasn't been already
    if not hasattr(universe, "listening_socket"):
        universe.initialize_server_socket()

    # assign a user if a new connection is waiting
    user = check_for_connection(universe.listening_socket)
    if user:
        universe.userlist.append(user)

    # iterate over the connected users
    for user in universe.userlist:
        user.pulse()

    # pause for a configurable amount of time (decimal seconds)
    time.sleep(universe.contents["mudpy.timing"].get("increment"))


def reload_data():
    """Reload all relevant objects."""
    universe.save()
    old_userlist = universe.userlist[:]
    old_loglines = universe.loglines[:]
    for element in list(universe.contents.values()):
        element.destroy()
    pending_loglines = universe.load()
    new_loglines = universe.loglines[:]
    universe.loglines = old_loglines + new_loglines + pending_loglines
    for user in old_userlist:
        user.reload()


def check_for_connection(listening_socket):
    """Check for a waiting connection and return a new user object."""

    # try to accept a new connection
    try:
        connection, address = listening_socket.accept()
    except BlockingIOError:
        return None

    # note that we got one
    log("New connection from %s." % address[0], 2)

    # disable blocking so we can proceed whether or not we can send/receive
    connection.setblocking(0)

    # create a new user object
    user = User()
    log("Instantiated %s for %s." % (user, address[0]), 0)

    # associate this connection with it
    user.connection = connection

    # set the user's ipa from the connection's ipa
    user.address = address[0]

    # let the client know we WILL EOR (RFC 885)
    mudpy.telnet.enable(user, mudpy.telnet.TELOPT_EOR, mudpy.telnet.US)
    user.negotiation_pause = 2

    # return the new user object
    return user


def find_command(command_name):
    """Try to find a command by name or abbreviation."""

    # lowercase the command
    command_name = command_name.lower()

    command = None
    if command_name in universe.groups["command"]:
        # the command matches a command word for which we have data
        command = universe.groups["command"][command_name]
    else:
        for candidate in sorted(universe.groups["command"]):
            if candidate.startswith(command_name) and not universe.groups[
                    "command"][candidate].is_restricted():
                # the command matches the start of a command word and is not
                # restricted to administrators
                command = universe.groups["command"][candidate]
                break
    return command


def get_menu(state, error=None, choices=None):
    """Show the correct menu text to a user."""

    # make sure we don't reuse a mutable sequence by default
    if choices is None:
        choices = {}

    # get the description or error text
    message = get_menu_description(state, error)

    # get menu choices for the current state
    message += get_formatted_menu_choices(state, choices)

    # try to get a prompt, if it was defined
    message += get_menu_prompt(state)

    # throw in the default choice, if it exists
    message += get_formatted_default_menu_choice(state)

    # display a message indicating if echo is off
    message += get_echo_message(state)

    # return the assembly of various strings defined above
    return message


def menu_echo_on(state):
    """True if echo is on, false if it is off."""
    return universe.groups["menu"][state].get("echo", True)


def get_echo_message(state):
    """Return a message indicating that echo is off."""
    if menu_echo_on(state):
        return ""
    else:
        return "(won't echo) "


def get_default_menu_choice(state):
    """Return the default choice for a menu."""
    return universe.groups["menu"][state].get("default")


def get_formatted_default_menu_choice(state):
    """Default menu choice foratted for inclusion in a prompt string."""
    default_choice = get_default_menu_choice(state)
    if default_choice:
        return "[$(red)" + default_choice + "$(nrm)] "
    else:
        return ""


def get_menu_description(state, error):
    """Get the description or error text."""

    # an error condition was raised by the handler
    if error:

        # try to get an error message matching the condition
        # and current state
        description = universe.groups[
            "menu"][state].get("error_" + error)
        if not description:
            description = "That is not a valid choice..."
        description = "$(red)" + description + "$(nrm)"

    # there was no error condition
    else:

        # try to get a menu description for the current state
        description = universe.groups["menu"][state].get("description")

    # return the description or error message
    if description:
        description += "$(eol)$(eol)"
    return description


def get_menu_prompt(state):
    """Try to get a prompt, if it was defined."""
    prompt = universe.groups["menu"][state].get("prompt")
    if prompt:
        prompt += " "
    return prompt


def get_menu_choices(user):
    """Return a dict of choice:meaning."""
    state = universe.groups["menu"][user.state]
    create_choices = state.get("create")
    if create_choices:
        choices = call_hook_function(create_choices, (user,))
    else:
        choices = {}
    ignores = []
    options = {}
    creates = {}
    for facet in state.facets():
        if facet.startswith("demand_") and not call_hook_function(
                universe.groups["menu"][user.state].get(facet), (user,)):
            ignores.append(facet.split("_", 2)[1])
        elif facet.startswith("create_"):
            creates[facet] = facet.split("_", 2)[1]
        elif facet.startswith("choice_"):
            options[facet] = facet.split("_", 2)[1]
    for facet in creates.keys():
        if not creates[facet] in ignores:
            choices[creates[facet]] = call_hook_function(
                state.get(facet), (user,))
    for facet in options.keys():
        if not options[facet] in ignores:
            choices[options[facet]] = state.get(facet)
    return choices


def get_formatted_menu_choices(state, choices):
    """Returns a formatted string of menu choices."""
    choice_output = ""
    choice_keys = list(choices.keys())
    choice_keys.sort()
    for choice in choice_keys:
        choice_output += "   [$(red)" + choice + "$(nrm)]  " + choices[
            choice
        ] + "$(eol)"
    if choice_output:
        choice_output += "$(eol)"
    return choice_output


def get_menu_branches(state):
    """Return a dict of choice:branch."""
    branches = {}
    for facet in universe.groups["menu"][state].facets():
        if facet.startswith("branch_"):
            branches[
                facet.split("_", 2)[1]
            ] = universe.groups["menu"][state].get(facet)
    return branches


def get_default_branch(state):
    """Return the default branch."""
    return universe.groups["menu"][state].get("branch")


def get_choice_branch(user):
    """Returns the new state matching the given choice."""
    branches = get_menu_branches(user.state)
    if user.choice in branches.keys():
        return branches[user.choice]
    elif user.choice in user.menu_choices.keys():
        return get_default_branch(user.state)
    else:
        return ""


def get_menu_actions(state):
    """Return a dict of choice:branch."""
    actions = {}
    for facet in universe.groups["menu"][state].facets():
        if facet.startswith("action_"):
            actions[
                facet.split("_", 2)[1]
            ] = universe.groups["menu"][state].get(facet)
    return actions


def get_default_action(state):
    """Return the default action."""
    return universe.groups["menu"][state].get("action")


def get_choice_action(user):
    """Run any indicated script for the given choice."""
    actions = get_menu_actions(user.state)
    if user.choice in actions.keys():
        return actions[user.choice]
    elif user.choice in user.menu_choices.keys():
        return get_default_action(user.state)
    else:
        return ""


def call_hook_function(fname, arglist):
    """Safely execute named function with supplied arguments, return result."""

    # all functions relative to mudpy package
    function = mudpy

    for component in fname.split("."):
        try:
            function = getattr(function, component)
        except AttributeError:
            log('Could not find mudpy.%s() for arguments "%s"'
                % (fname, arglist), 7)
            function = None
            break
    if function:
        try:
            return function(*arglist)
        except Exception:
            log('Calling mudpy.%s(%s) raised an exception...\n%s'
                % (fname, (*arglist,), traceback.format_exc()), 7)


def handle_user_input(user):
    """The main handler, branches to a state-specific handler."""

    # if the user's client echo is off, send a blank line for aesthetics
    if mudpy.telnet.is_enabled(user, mudpy.telnet.TELOPT_ECHO,
                               mudpy.telnet.US):
        user.send("", add_prompt=False, prepend_padding=False)

    # check to make sure the state is expected, then call that handler
    try:
        globals()["handler_" + user.state](user)
    except KeyError:
        generic_menu_handler(user)

    # since we got input, flag that the menu/prompt needs to be redisplayed
    user.menu_seen = False

    # update the last_input timestamp while we're at it
    user.last_input = universe.get_time()


def generic_menu_handler(user):
    """A generic menu choice handler."""

    # get a lower-case representation of the next line of input
    if user.input_queue:
        user.choice = user.input_queue.pop(0)
        if user.choice:
            user.choice = user.choice.lower()
    else:
        user.choice = ""
    if not user.choice:
        user.choice = get_default_menu_choice(user.state)
    if user.choice in user.menu_choices:
        action = get_choice_action(user)
        if action:
            call_hook_function(action, (user,))
        new_state = get_choice_branch(user)
        if new_state:
            user.state = new_state
    else:
        user.error = "default"


def handler_entering_account_name(user):
    """Handle the login account name."""

    # get the next waiting line of input
    input_data = user.input_queue.pop(0)

    # did the user enter anything?
    if input_data:

        # keep only the first word and convert to lower-case
        name = input_data.lower()

        # fail if there are non-alphanumeric characters
        if name != "".join(filter(
                lambda x: x >= "0" and x <= "9" or x >= "a" and x <= "z",
                name)):
            user.error = "bad_name"

        # if that account exists, time to request a password
        elif name in universe.groups.get("account", {}):
            user.account = universe.groups["account"][name]
            user.state = "checking_password"

        # otherwise, this could be a brand new user
        else:
            user.account = Element("account.%s" % name, universe)
            user.account.set("name", name)
            log("New user: " + name, 2)
            user.state = "checking_new_account_name"

    # if the user entered nothing for a name, then buhbye
    else:
        user.state = "disconnecting"


def handler_checking_password(user):
    """Handle the login account password."""

    # get the next waiting line of input
    input_data = user.input_queue.pop(0)

    if "mudpy.limit" in universe.contents:
        max_password_tries = universe.contents["mudpy.limit"].get(
            "password_tries", 3)
    else:
        max_password_tries = 3

    # does the hashed input equal the stored hash?
    if mudpy.password.verify(input_data, user.account.get("passhash")):

        # if so, set the username and load from cold storage
        if not user.replace_old_connections():
            user.authenticate()
            user.state = "main_utility"

    # if at first your hashes don't match, try, try again
    elif user.password_tries < max_password_tries - 1:
        user.password_tries += 1
        user.error = "incorrect"

    # we've exceeded the maximum number of password failures, so disconnect
    else:
        user.send(
            "$(eol)$(red)Too many failed password attempts...$(nrm)$(eol)"
        )
        user.state = "disconnecting"


def handler_entering_new_password(user):
    """Handle a new password entry."""

    # get the next waiting line of input
    input_data = user.input_queue.pop(0)

    if "mudpy.limit" in universe.contents:
        max_password_tries = universe.contents["mudpy.limit"].get(
            "password_tries", 3)
    else:
        max_password_tries = 3

    # make sure the password is strong--at least one upper, one lower and
    # one digit, seven or more characters in length
    if len(input_data) > 6 and len(
       list(filter(lambda x: x >= "0" and x <= "9", input_data))
       ) and len(
        list(filter(lambda x: x >= "A" and x <= "Z", input_data))
    ) and len(
        list(filter(lambda x: x >= "a" and x <= "z", input_data))
    ):

        # hash and store it, then move on to verification
        user.account.set("passhash", mudpy.password.create(input_data))
        user.state = "verifying_new_password"

    # the password was weak, try again if you haven't tried too many times
    elif user.password_tries < max_password_tries - 1:
        user.password_tries += 1
        user.error = "weak"

    # too many tries, so adios
    else:
        user.send(
            "$(eol)$(red)Too many failed password attempts...$(nrm)$(eol)"
        )
        user.account.destroy()
        user.state = "disconnecting"


def handler_verifying_new_password(user):
    """Handle the re-entered new password for verification."""

    # get the next waiting line of input
    input_data = user.input_queue.pop(0)

    if "mudpy.limit" in universe.contents:
        max_password_tries = universe.contents["mudpy.limit"].get(
            "password_tries", 3)
    else:
        max_password_tries = 3

    # hash the input and match it to storage
    if mudpy.password.verify(input_data, user.account.get("passhash")):
        user.authenticate()

        # the hashes matched, so go active
        if not user.replace_old_connections():
            user.state = "main_utility"

    # go back to entering the new password as long as you haven't tried
    # too many times
    elif user.password_tries < max_password_tries - 1:
        user.password_tries += 1
        user.error = "differs"
        user.state = "entering_new_password"

    # otherwise, sayonara
    else:
        user.send(
            "$(eol)$(red)Too many failed password attempts...$(nrm)$(eol)"
        )
        user.account.destroy()
        user.state = "disconnecting"


def handler_active(user):
    """Handle input for active users."""

    # get the next waiting line of input
    input_data = user.input_queue.pop(0)

    # is there input?
    if input_data:

        # split out the command and parameters
        actor = user.avatar
        mode = actor.get("mode")
        if mode and input_data.startswith("!"):
            command_name, parameters = first_word(input_data[1:])
        elif mode == "chat":
            command_name = "say"
            parameters = input_data
        else:
            command_name, parameters = first_word(input_data)

        # expand to an actual command
        command = find_command(command_name)

        # if it's allowed, do it
        result = None
        if actor.can_run(command):
            action_fname = command.get("action", command.key)
            if action_fname:
                result = call_hook_function(action_fname, (actor, parameters))

        # if the command was not run, give an error
        if not result:
            mudpy.command.error(actor, input_data)

    # if no input, just idle back with a prompt
    else:
        user.send("", just_prompt=True)


def daemonize(universe):
    """Fork and disassociate from everything."""

    # only if this is what we're configured to do
    if "mudpy.process" in universe.contents and universe.contents[
            "mudpy.process"].get("daemon"):

        # log before we start forking around, so the terminal gets the message
        log("Disassociating from the controlling terminal.")

        # fork off and die, so we free up the controlling terminal
        if os.fork():
            os._exit(0)

        # switch to a new process group
        os.setsid()

        # fork some more, this time to free us from the old process group
        if os.fork():
            os._exit(0)

        # reset the working directory so we don't needlessly tie up mounts
        os.chdir("/")

        # clear the file creation mask so we can bend it to our will later
        os.umask(0)

        # redirect stdin/stdout/stderr and close off their former descriptors
        for stdpipe in range(3):
            os.close(stdpipe)
        sys.stdin = codecs.open("/dev/null", "r", "utf-8")
        sys.__stdin__ = codecs.open("/dev/null", "r", "utf-8")
        sys.stdout = codecs.open("/dev/null", "w", "utf-8")
        sys.stderr = codecs.open("/dev/null", "w", "utf-8")
        sys.__stdout__ = codecs.open("/dev/null", "w", "utf-8")
        sys.__stderr__ = codecs.open("/dev/null", "w", "utf-8")


def create_pidfile(universe):
    """Write a file containing the current process ID."""
    pid = str(os.getpid())
    log("Process ID: " + pid)
    if "mudpy.process" in universe.contents:
        file_name = universe.contents["mudpy.process"].get("pidfile", "")
    else:
        file_name = ""
    if file_name:
        if not os.path.isabs(file_name):
            file_name = os.path.join(universe.startdir, file_name)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        file_descriptor = codecs.open(file_name, "w", "utf-8")
        file_descriptor.write(pid + "\n")
        file_descriptor.flush()
        file_descriptor.close()


def remove_pidfile(universe):
    """Remove the file containing the current process ID."""
    if "mudpy.process" in universe.contents:
        file_name = universe.contents["mudpy.process"].get("pidfile", "")
    else:
        file_name = ""
    if file_name:
        if not os.path.isabs(file_name):
            file_name = os.path.join(universe.startdir, file_name)
        if os.access(file_name, os.W_OK):
            os.remove(file_name)


def excepthook(excepttype, value, tracebackdata):
    """Handle uncaught exceptions."""

    # assemble the list of errors into a single string
    message = "".join(
        traceback.format_exception(excepttype, value, tracebackdata)
    )

    # try to log it, if possible
    try:
        log(message, 9)
    except Exception as e:
        # try to write it to stderr, if possible
        sys.stderr.write(message + "\nException while logging...\n%s" % e)


def sighook(what, where):
    """Handle external signals."""

    # a generic message
    message = "Caught signal: "

    # for a hangup signal
    if what == signal.SIGHUP:
        message += "hangup (reloading)"
        universe.reload_flag = True

    # for a terminate signal
    elif what == signal.SIGTERM:
        message += "terminate (halting)"
        universe.terminate_flag = True

    # catchall for unexpected signals
    else:
        message += str(what) + " (unhandled)"

    # log what happened
    log(message, 8)


def override_excepthook():
    """Redefine sys.excepthook with our own."""
    sys.excepthook = excepthook


def assign_sighook():
    """Assign a customized handler for some signals."""
    signal.signal(signal.SIGHUP, sighook)
    signal.signal(signal.SIGTERM, sighook)


def setup():
    """This contains functions to be performed when starting the engine."""

    # see if a configuration file was specified
    if len(sys.argv) > 1:
        conffile = sys.argv[1]
    else:
        conffile = ""

    # the big bang
    global universe
    universe = Universe(conffile, True)

    # report any loglines which accumulated during setup
    for logline in universe.setup_loglines:
        log(*logline)
    universe.setup_loglines = []

    # fork and disassociate
    daemonize(universe)

    # override the default exception handler so we get logging first thing
    override_excepthook()

    # set up custom signal handlers
    assign_sighook()

    # make the pidfile
    create_pidfile(universe)

    # load and store diagnostic info
    universe.versions = mudpy.version.Versions("mudpy")

    # log startup diagnostic messages
    log("On %s at %s" % (universe.versions.python_version, sys.executable), 1)
    log("Import path: %s" % ", ".join(sys.path), 1)
    log("Installed dependencies: %s" % universe.versions.dependencies_text, 1)
    log("Other python packages: %s" % universe.versions.environment_text, 1)
    log("Running version: %s" % universe.versions.version, 1)
    log("Initial directory: %s" % universe.startdir, 1)
    log("Command line: %s" % " ".join(sys.argv), 1)

    # pass the initialized universe back
    return universe


def finish():
    """These are functions performed when shutting down the engine."""

    # the loop has terminated, so save persistent data
    universe.save()

    # log a final message
    log("Shutting down now.")

    # get rid of the pidfile
    remove_pidfile(universe)
