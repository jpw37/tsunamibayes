"""User command functions for the mudpy engine."""

# Copyright (c) 2004-2020 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

import random
import re
import traceback
import unicodedata

import mudpy


def chat(actor, parameters):
    """Toggle chat mode."""
    mode = actor.get("mode")
    if not mode:
        actor.set("mode", "chat")
        actor.send("Entering chat mode (use $(grn)!chat$(nrm) to exit).")
    elif mode == "chat":
        actor.remove_facet("mode")
        actor.send("Exiting chat mode.")
    else:
        actor.send("Sorry, but you're already busy with something else!")
    return True


def create(actor, parameters):
    """Create an element if it does not exist."""
    if not parameters:
        message = "You must at least specify an element to create."
    elif not actor.owner:
        message = ""
    else:
        arguments = parameters.split()
        if len(arguments) == 1:
            arguments.append("")
        if len(arguments) == 2:
            element, filename = arguments
            if element in actor.universe.contents:
                message = 'The "' + element + '" element already exists.'
            else:
                message = ('You create "' +
                           element + '" within the universe.')
                logline = actor.owner.account.get(
                    "name"
                ) + " created an element: " + element
                if filename:
                    logline += " in file " + filename
                    if filename not in actor.universe.files:
                        message += (
                            ' Warning: "' + filename + '" is not yet '
                            "included in any other file and will not be read "
                            "on startup unless this is remedied.")
                mudpy.misc.Element(element, actor.universe, filename)
                mudpy.misc.log(logline, 6)
        elif len(arguments) > 2:
            message = "You can only specify an element and a filename."
    actor.send(message)
    return True


def delete(actor, parameters):
    """Delete a facet from an element."""
    if not parameters:
        message = "You must specify an element and a facet."
    else:
        arguments = parameters.split(" ")
        if len(arguments) == 1:
            message = ('What facet of element "' + arguments[0]
                       + '" would you like to delete?')
        elif len(arguments) != 2:
            message = "You may only specify an element and a facet."
        else:
            element, facet = arguments
            if element not in actor.universe.contents:
                message = 'The "' + element + '" element does not exist.'
            elif facet not in actor.universe.contents[element].facets():
                message = ('The "' + element + '" element has no "' + facet
                           + '" facet.')
            else:
                actor.universe.contents[element].remove_facet(facet)
                message = ('You have successfully deleted the "' + facet
                           + '" facet of element "' + element
                           + '". Try "show element ' +
                           element + '" for verification.')
    actor.send(message)
    return True


def destroy(actor, parameters):
    """Destroy an element if it exists."""
    if actor.owner:
        if not parameters:
            message = "You must specify an element to destroy."
        else:
            if parameters not in actor.universe.contents:
                message = 'The "' + parameters + '" element does not exist.'
            else:
                actor.universe.contents[parameters].destroy()
                message = ('You destroy "' + parameters
                           + '" within the universe.')
                mudpy.misc.log(
                    actor.owner.account.get(
                        "name"
                    ) + " destroyed an element: " + parameters,
                    6
                )
        actor.send(message)
    return True


def error(actor, input_data):
    """Generic error for an unrecognized command word."""

    # 90% of the time use a generic error
    # Allow the random.randrange() call in bandit since it's not used for
    # security/cryptographic purposes
    if random.randrange(10):  # nosec
        message = '''I'm not sure what "''' + input_data + '''" means...'''

    # 10% of the time use the classic diku error
    else:
        message = "Arglebargle, glop-glyf!?!"

    # try to send the error message, and log if we can't
    try:
        actor.send(message)
    except Exception:
        mudpy.misc.log(
            'Sending a command error to user %s raised exception...\n%s' % (
                actor.owner.account.get("name"), traceback.format_exc()))
    return True


def evaluate(actor, parameters):
    """Evaluate a Python expression."""

    if not parameters:
        message = "You need to supply a Python expression."
    elif "__" in parameters:
        message = "Double-underscores (__) are not allowed in expressions."
    elif "lambda" in parameters:
        message = "Lambda functions are not allowed in expressions."
    else:
        # Strictly limit the allowed builtins and modules
        eval_globals = {"__builtins__": dict()}
        for allowed in ("dir", "globals", "len", "locals"):
            eval_globals["__builtins__"][allowed] = __builtins__[allowed]
        eval_globals["mudpy"] = mudpy
        eval_globals["universe"] = actor.universe
        try:
            # there is no other option than to use eval() for this, since
            # its purpose is to evaluate arbitrary expressions, so do what
            # we can to secure it and allow it for bandit analysis
            message = repr(eval(parameters, eval_globals))  # nosec
        except Exception as e:
            message = ("$(red)Your expression raised an exception...$(eol)"
                       "$(eol)$(bld)%s$(nrm)" % e)
    actor.send(message)
    return True


def halt(actor, parameters):
    """Halt the world."""
    if actor.owner:

        # see if there's a message or use a generic one
        if parameters:
            message = "Halting: " + parameters
        else:
            message = "User " + actor.owner.account.get(
                "name"
            ) + " halted the world."

        # let everyone know
        mudpy.misc.broadcast(message, add_prompt=False)
        mudpy.misc.log(message, 8)

        # set a flag to terminate the world
        actor.universe.terminate_flag = True
    return True


def help(actor, parameters):
    """List available commands and provide help for commands."""

    # did the user ask for help on a specific command word?
    if parameters and actor.owner:

        # is the command word one for which we have data?
        command = mudpy.misc.find_command(parameters)

        # only for allowed commands
        if actor.can_run(command):

            # add a description if provided
            description = command.get("description")
            if not description:
                description = "(no short description provided)"
            if command.is_restricted():
                output = "$(red)"
            else:
                output = "$(grn)"
            output = "%s%s$(nrm) - %s$(eol)$(eol)" % (
                output, command.subkey, description)

            # add the help text if provided
            help_text = command.get("help")
            if not help_text:
                help_text = "No help is provided for this command."
            output += help_text

            # list related commands
            see_also = command.get("see_also")
            if see_also:
                really_see_also = ""
                for item in see_also:
                    if item in actor.universe.groups["command"]:
                        command = actor.universe.groups["command"][item]
                        if actor.can_run(command):
                            if really_see_also:
                                really_see_also += ", "
                            if command.is_restricted():
                                really_see_also += "$(red)"
                            else:
                                really_see_also += "$(grn)"
                            really_see_also += item + "$(nrm)"
                if really_see_also:
                    output += "$(eol)$(eol)See also: " + really_see_also

        # no data for the requested command word
        else:
            output = "That is not an available command."

    # no specific command word was indicated
    else:

        # preamble text
        output = ("These are the commands available to you [brackets indicate "
                  "optional portion]:$(eol)$(eol)")

        # list command names in alphabetical order
        for command_name, command in sorted(
                actor.universe.groups["command"].items()):

            # skip over disallowed commands
            if actor.can_run(command):

                # start incrementing substrings
                for position in range(1, len(command_name) + 1):

                    # we've found our shortest possible abbreviation
                    candidate = mudpy.misc.find_command(
                            command_name[:position])
                    try:
                        if candidate.subkey == command_name:
                            break
                    except AttributeError:
                        pass

                # use square brackets to indicate optional part of command name
                if position < len(command_name):
                    abbrev = "%s[%s]" % (
                        command_name[:position], command_name[position:])
                else:
                    abbrev = command_name

                # supply a useful default if the short description is missing
                description = command.get(
                    "description", "(no short description provided)")

                # administrative command names are in red, others in green
                if command.is_restricted():
                    color = "red"
                else:
                    color = "grn"

                # format the entry for this command
                output = "%s   $(%s)%s$(nrm) - %s$(eol)" % (
                    output, color, abbrev, description)

        # add a footer with instructions on getting additional information
        output = ('%s $(eol)Enter "help COMMAND" for help on a command named '
                  '"COMMAND".' % output)

    # send the accumulated output to the user
    actor.send(output)
    return True


def look(actor, parameters):
    """Look around."""
    if parameters:
        actor.send("You can't look at or in anything yet.")
    else:
        actor.look_at(actor.get("location"))
    return True


def move(actor, parameters):
    """Move the avatar in a given direction."""
    for portal in sorted(
            actor.universe.contents[actor.get("location")].portals()):
        if portal.startswith(parameters):
            actor.move_direction(portal)
            return portal
    actor.send("You cannot go that way.")
    return True


def preferences(actor, parameters):
    """List, view and change actor preferences."""

    # Escape replacement macros in preferences
    parameters = mudpy.misc.escape_macros(parameters)

    message = ""
    arguments = parameters.split()
    allowed_prefs = set()
    base_prefs = []
    user_config = actor.universe.contents.get("mudpy.user")
    if user_config:
        base_prefs = user_config.get("pref_allow", [])
        allowed_prefs.update(base_prefs)
        if actor.owner.account.get("administrator"):
            allowed_prefs.update(user_config.get("pref_admin", []))
    if not arguments:
        message += "These are your current preferences:"

        # color-code base and admin prefs
        for pref in sorted(allowed_prefs):
            if pref in base_prefs:
                color = "grn"
            else:
                color = "red"
            message += ("$(eol)   $(%s)%s$(nrm) - %s" % (
                color, pref, actor.owner.account.get(pref, "<not set>")))

    elif arguments[0] not in allowed_prefs:
        message += (
            'Preference "%s" does not exist. Try the `preferences` command by '
            "itself for a list of valid preferences." % arguments[0])
    elif len(arguments) == 1:
        message += "%s" % actor.owner.account.get(arguments[0], "<not set>")
    else:
        pref = arguments[0]
        value = " ".join(arguments[1:])
        try:
            actor.owner.account.set(pref, value)
            message += 'Preference "%s" set to "%s".' % (pref, value)
        except ValueError:
            message = (
                'Preference "%s" cannot be set to type "%s".' % (
                    pref, type(value)))
    actor.send(message)
    return True


def quit(actor, parameters):
    """Leave the world and go back to the main menu."""
    if actor.owner:
        actor.owner.state = "main_utility"
        actor.owner.deactivate_avatar()
    return True


def reload(actor, parameters):
    """Reload all code modules, configs and data."""
    if actor.owner:

        # let the user know and log
        actor.send("Reloading all code modules, configs and data.")
        mudpy.misc.log(
            "User " +
            actor.owner.account.get("name") + " reloaded the world.",
            6
        )

        # set a flag to reload
        actor.universe.reload_flag = True
    return True


def say(actor, parameters):
    """Speak to others in the same area."""

    # check for replacement macros and escape them
    parameters = mudpy.misc.escape_macros(parameters)

    # if the message is wrapped in quotes, remove them and leave contents
    # intact
    if parameters.startswith('"') and parameters.endswith('"'):
        message = parameters[1:-1]
        literal = True

    # otherwise, get rid of stray quote marks on the ends of the message
    else:
        message = parameters.strip('''"'`''')
        literal = False

    # the user entered a message
    if message:

        # match the punctuation used, if any, to an action
        if "mudpy.linguistic" in actor.universe.contents:
            actions = actor.universe.contents[
                "mudpy.linguistic"].get("actions", {})
            default_punctuation = (actor.universe.contents[
                "mudpy.linguistic"].get("default_punctuation", "."))
        else:
            actions = {}
            default_punctuation = "."
        action = ""

        # reverse sort punctuation options so the longest match wins
        for mark in sorted(actions.keys(), reverse=True):
            if not literal and message.endswith(mark):
                action = actions[mark]
                break

        # add punctuation if needed
        if not action:
            action = actions[default_punctuation]
            if message and not (
               literal or unicodedata.category(message[-1]) == "Po"
               ):
                message += default_punctuation

        # failsafe checks to avoid unwanted reformatting and null strings
        if message and not literal:

            # decapitalize the first letter to improve matching
            message = message[0].lower() + message[1:]

            # iterate over all words in message, replacing typos
            if "mudpy.linguistic" in actor.universe.contents:
                typos = actor.universe.contents[
                    "mudpy.linguistic"].get("typos", {})
            else:
                typos = {}
            words = message.split()
            for index in range(len(words)):
                word = words[index]
                while unicodedata.category(word[0]) == "Po":
                    word = word[1:]
                while unicodedata.category(word[-1]) == "Po":
                    word = word[:-1]
                if word in typos.keys():
                    words[index] = words[index].replace(word, typos[word])
            message = " ".join(words)

            # capitalize the first letter
            message = message[0].upper() + message[1:]

    # tell the area
    if message:
        actor.echo_to_location(
            actor.get("name") + " " + action + 's, "' + message + '"'
        )
        actor.send("You " + action + ', "' + message + '"')

    # there was no message
    else:
        actor.send("What do you want to say?")
    return True


def c_set(actor, parameters):
    """Set a facet of an element."""
    if not parameters:
        message = "You must specify an element, a facet and a value."
    else:
        arguments = parameters.split(" ", 2)
        if len(arguments) == 1:
            message = ('What facet of element "' + arguments[0]
                       + '" would you like to set?')
        elif len(arguments) == 2:
            message = ('What value would you like to set for the "' +
                       arguments[1] + '" facet of the "' + arguments[0]
                       + '" element?')
        else:
            element, facet, value = arguments
            if element not in actor.universe.contents:
                message = 'The "' + element + '" element does not exist.'
            else:
                try:
                    actor.universe.contents[element].set(facet, value)
                except PermissionError:
                    message = ('The "%s" element is kept in read-only file '
                               '"%s" and cannot be altered.' %
                               (element, actor.universe.contents[
                                        element].origin.source))
                except ValueError:
                    message = ('Value "%s" of type "%s" cannot be coerced '
                               'to the correct datatype for facet "%s".' %
                               (value, type(value), facet))
                else:
                    message = ('You have successfully (re)set the "' + facet
                               + '" facet of element "' + element
                               + '". Try "show element ' +
                               element + '" for verification.')
    actor.send(message)
    return True


def show(actor, parameters):
    """Show program data."""
    message = ""
    arguments = parameters.split()
    if not parameters:
        message = "What do you want to show?"
    elif arguments[0] == "version":
        message = repr(actor.universe.versions)
    elif arguments[0] == "time":
        message = "%s increments elapsed since the world was created." % (
            str(actor.universe.groups["internal"]["counters"].get("elapsed")))
    elif arguments[0] == "groups":
        message = "These are the element groups:$(eol)"
        groups = list(actor.universe.groups.keys())
        groups.sort()
        for group in groups:
            message += "$(eol)   $(grn)" + group + "$(nrm)"
    elif arguments[0] == "files":
        message = "These are the current files containing the universe:$(eol)"
        filenames = sorted(actor.universe.files)
        for filename in filenames:
            if actor.universe.files[filename].is_writeable():
                status = "rw"
            else:
                status = "ro"
            message += ("$(eol)   $(red)(%s) $(grn)%s$(nrm)" %
                        (status, filename))
            if actor.universe.files[filename].flags:
                message += (" $(yel)[%s]$(nrm)" %
                            ",".join(actor.universe.files[filename].flags))
    elif arguments[0] == "group":
        if len(arguments) != 2:
            message = "You must specify one group."
        elif arguments[1] in actor.universe.groups:
            message = ('These are the elements in the "' + arguments[1]
                       + '" group:$(eol)')
            elements = [
                (
                    actor.universe.groups[arguments[1]][x].key
                ) for x in actor.universe.groups[arguments[1]].keys()
            ]
            elements.sort()
            for element in elements:
                message += "$(eol)   $(grn)" + element + "$(nrm)"
        else:
            message = 'Group "' + arguments[1] + '" does not exist.'
    elif arguments[0] == "file":
        if len(arguments) != 2:
            message = "You must specify one file."
        elif arguments[1] in actor.universe.files:
            message = ('These are the nodes in the "' + arguments[1]
                       + '" file:$(eol)')
            elements = sorted(actor.universe.files[arguments[1]].data)
            for element in elements:
                message += "$(eol)   $(grn)" + element + "$(nrm)"
        else:
            message = 'File "%s" does not exist.' % arguments[1]
    elif arguments[0] == "element":
        if len(arguments) != 2:
            message = "You must specify one element."
        elif arguments[1].strip(".") in actor.universe.contents:
            element = actor.universe.contents[arguments[1].strip(".")]
            message = ('These are the properties of the "' + arguments[1]
                       + '" element (in "' + element.origin.source
                       + '"):$(eol)')
            facets = element.facets()
            for facet in sorted(facets):
                message += ("$(eol)   $(grn)%s: $(red)%s$(nrm)" %
                            (facet, str(facets[facet])))
        else:
            message = 'Element "' + arguments[1] + '" does not exist.'
    elif arguments[0] == "log":
        if len(arguments) == 4:
            if re.match(r"^\d+$", arguments[3]) and int(arguments[3]) >= 0:
                stop = int(arguments[3])
            else:
                stop = -1
        else:
            stop = 0
        if len(arguments) >= 3:
            if re.match(r"^\d+$", arguments[2]) and int(arguments[2]) > 0:
                start = int(arguments[2])
            else:
                start = -1
        else:
            start = 10
        if len(arguments) >= 2:
            if (re.match(r"^\d+$", arguments[1])
                    and 0 <= int(arguments[1]) <= 9):
                level = int(arguments[1])
            else:
                level = -1
        elif 0 <= actor.owner.account.get("loglevel", 0) <= 9:
            level = actor.owner.account.get("loglevel", 0)
        else:
            level = 1
        if level > -1 and start > -1 and stop > -1:
            message = mudpy.misc.get_loglines(level, start, stop)
        else:
            message = ("When specified, level must be 0-9 (default 1), "
                       "start and stop must be >=1 (default 10 and 1).")
    else:
        message = '''I don't know what "''' + parameters + '" is.'
    actor.send(message)
    return True
