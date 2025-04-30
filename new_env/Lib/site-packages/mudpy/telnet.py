"""Telnet functions and constants for the mudpy engine."""

# Copyright (c) 2004-2020 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

import mudpy

# telnet options (from bsd's arpa/telnet.h since telnetlib's are ambiguous)
TELOPT_BINARY = 0  # transmit 8-bit data by the receiver (rfc 856)
TELOPT_ECHO = 1  # echo received data back to the sender (rfc 857)
TELOPT_SGA = 3  # suppress transmission of the go ahead character (rfc 858)
TELOPT_TTYPE = 24  # exchange terminal type information (rfc 1091)
TELOPT_EOR = 25  # transmit end-of-record after transmitting data (rfc 885)
TELOPT_NAWS = 31  # negotiate about window size (rfc 1073)
TELOPT_LINEMODE = 34  # cooked line-by-line input mode (rfc 1184)
option_names = {
    TELOPT_BINARY: "8-bit binary",
    TELOPT_ECHO: "echo",
    TELOPT_SGA: "suppress go-ahead",
    TELOPT_TTYPE: "terminal type",
    TELOPT_EOR: "end of record",
    TELOPT_NAWS: "negotiate about window size",
    TELOPT_LINEMODE: "line mode",
}

supported = (
    TELOPT_BINARY,
    TELOPT_ECHO,
    TELOPT_SGA,
    TELOPT_TTYPE,
    TELOPT_EOR,
    TELOPT_NAWS,
    TELOPT_LINEMODE
)

# telnet commands
EOR = 239  # end-of-record signal (rfc 885)
SE = 240  # end of subnegotiation parameters (rfc 854)
GA = 249  # go ahead signal (rfc 854)
SB = 250  # what follows is subnegotiation of the indicated option (rfc 854)
WILL = 251  # desire or confirmation of performing an option (rfc 854)
WONT = 252  # refusal or confirmation of performing an option (rfc 854)
DO = 253  # request or confirm performing an option (rfc 854)
DONT = 254  # demand or confirm no longer performing an option (rfc 854)
IAC = 255  # interpret as command escape character (rfc 854)
command_names = {
    EOR: "end of record",
    SE: "subnegotiation end",
    GA: "go ahead",
    SB: "subnegotiation begin",
    WILL: "will",
    WONT: "won't",
    DO: "do",
    DONT: "don't",
    IAC: "interpret as command",
}

# RFC 1143 option negotiation states
NO = 0  # option is disabled
YES = 1  # option is enabled
WANTNO = 2  # demanded disabling option
WANTYES = 3  # requested enabling option
WANTNO_OPPOSITE = 4  # demanded disabling option but queued an enable after it
WANTYES_OPPOSITE = 5  # requested enabling option but queued a disable after it
state_names = {
    NO: "disabled",
    YES: "enabled",
    WANTNO: "demand disabling",
    WANTYES: "request enabling",
    WANTNO_OPPOSITE: "want no queued opposite",
    WANTYES_OPPOSITE: "want yes queued opposite",
}

# RFC 1143 option negotiation parties
HIM = 0
US = 1
party_names = {
    HIM: "him",
    US: "us",
}

# RFC 1091 commands
IS = 0
SEND = 1
ttype_command_names = {
    IS: "is",
    SEND: "send",
}


def log(message, user):
    """Log debugging info for Telnet client/server interactions."""
    if user.account:
        client = user.account.get("name", user)
    else:
        client = user
    mudpy.misc.log('[telnet] %s %s.' % (message, client), 0)


def telnet_proto(*arguments):
    """Return a concatenated series of Telnet protocol commands."""
    return bytes((arguments))


def translate_action(*command):
    """Convert a Telnet command sequence into text suitable for logging."""
    try:
        command_name = command_names[command[0]]
    except KeyError:
        # This should never happen since we filter unknown commands from
        # the input queue, but added here for completeness since logging
        # should never crash the process
        command_name = str(command[0])
    try:
        option_name = option_names[command[1]]
    except KeyError:
        # This can happen for any of the myriad of Telnet options missing
        # from the option_names dict
        option_name = str(command[1])
    return "%s %s" % (command_name, option_name)


def send_command(user, *command):
    """Sends a Telnet command string to the specified user's socket."""
    user.send(telnet_proto(IAC, *command), raw=True)
    log('Sent "%s" to' % translate_action(*command), user)


def is_enabled(user, telopt, party, state=YES):
    """Indicates whether a specified Telnet option is enabled."""
    if (telopt, party) in user.telopts and user.telopts[
       (telopt, party)
       ] is state:
        return True
    else:
        return False


def enable(user, telopt, party):
    """Negotiates enabling a Telnet option for the indicated user's socket."""
    if party is HIM:
        txpos = DO
    else:
        txpos = WILL
    if not (telopt, party) in user.telopts or user.telopts[
       (telopt, party)
       ] is NO:
        user.telopts[(telopt, party)] = WANTYES
        send_command(user, txpos, telopt)
    elif user.telopts[(telopt, party)] is WANTNO:
        user.telopts[(telopt, party)] = WANTNO_OPPOSITE
    elif user.telopts[(telopt, party)] is WANTYES_OPPOSITE:
        user.telopts[(telopt, party)] = WANTYES


def disable(user, telopt, party):
    """Negotiates disabling a Telnet option for the user's socket."""
    if party is HIM:
        txneg = DONT
    else:
        txneg = WONT
    if not (telopt, party) in user.telopts:
        user.telopts[(telopt, party)] = NO
    elif user.telopts[(telopt, party)] is YES:
        user.telopts[(telopt, party)] = WANTNO
        send_command(user, txneg, telopt)
    elif user.telopts[(telopt, party)] is WANTYES:
        user.telopts[(telopt, party)] = WANTYES_OPPOSITE
    elif user.telopts[(telopt, party)] is WANTNO_OPPOSITE:
        user.telopts[(telopt, party)] = WANTNO


def request_ttype(user):
    """Clear and request the terminal type."""

    # only actually request if the corresponding telopt is enabled
    if is_enabled(user, TELOPT_TTYPE, HIM):
        # set to the empty string to indicate it's been requested
        user.ttype = ""
        user.send(telnet_proto(IAC, SB, TELOPT_TTYPE, SEND, IAC, SE), raw=True)
        log('Sent terminal type request to', user)


def negotiate_telnet_options(user):
    """Reply to and remove telnet negotiation options from partial_input."""

    # make a local copy to play with
    text = user.partial_input

    # start at the beginning of the input
    position = 0

    # as long as we haven't checked it all
    len_text = len(text)
    while position < len_text:

        # jump to the first IAC you find
        position = text.find(telnet_proto(IAC), position)

        # if there wasn't an IAC in the input or it's at the end, we're done
        if position < 0 or position >= len_text - 1:
            break

        # the byte following the IAC is our command
        command = text[position+1]

        # replace a double (literal) IAC if there's a CR+NUL or CR+LF later
        if command is IAC:
            if (
                    text.find(b"\r\0", position) > 0 or
                    text.find(b"\r\n", position) > 0):
                position += 1
                text = text[:position] + text[position + 1:]
                log('Escaped IAC from', user)
            else:
                position += 2

        # implement an RFC 1143 option negotiation queue here
        elif len_text > position + 2 and WILL <= command <= DONT:
            telopt = text[position+2]
            log('Received "%s" from' % translate_action(command, telopt), user)
            if telopt in supported:
                if command <= WONT:
                    party = HIM
                    rxpos = WILL
                    txpos = DO
                    txneg = DONT
                else:
                    party = US
                    rxpos = DO
                    txpos = WILL
                    txneg = WONT
                if (telopt, party) not in user.telopts:
                    user.telopts[(telopt, party)] = NO
                if command is rxpos:
                    if user.telopts[(telopt, party)] is NO:
                        user.telopts[(telopt, party)] = YES
                        send_command(user, txpos, telopt)
                    elif user.telopts[(telopt, party)] is WANTNO:
                        user.telopts[(telopt, party)] = NO
                    elif user.telopts[(telopt, party)] is WANTNO_OPPOSITE:
                        user.telopts[(telopt, party)] = YES
                    elif user.telopts[(telopt, party)] is WANTYES_OPPOSITE:
                        user.telopts[(telopt, party)] = WANTNO
                        send_command(user, txneg, telopt)
                    else:
                        user.telopts[(telopt, party)] = YES
                else:
                    if user.telopts[(telopt, party)] is YES:
                        user.telopts[(telopt, party)] = NO
                        send_command(user, txneg, telopt)
                    elif user.telopts[(telopt, party)] is WANTNO_OPPOSITE:
                        user.telopts[(telopt, party)] = WANTYES
                        send_command(user, txpos, telopt)
                    else:
                        user.telopts[(telopt, party)] = NO
            elif command is WILL:
                send_command(user, DONT, telopt)
            else:
                send_command(user, WONT, telopt)
            text = text[:position] + text[position + 3:]

        # subnegotiation options
        elif len_text > position + 4 and command is SB:
            telopt = text[position + 2]
            end_subnegotiation = text.find(telnet_proto(IAC, SE), position)
            if end_subnegotiation > 0:
                if telopt is TELOPT_NAWS:
                    user.columns = (
                        text[position + 3] * 256 + text[position + 4])
                    user.rows = (
                        text[position + 5] * 256 + text[position + 6])
                elif telopt is TELOPT_TTYPE and text[position + 3] is IS:
                    user.ttype = (
                        text[position + 4:end_subnegotiation]).decode("ascii")
                text = text[:position] + text[end_subnegotiation + 2:]
            else:
                position += 1

        # otherwise, strip out a two-byte IAC command
        elif len_text > position + 2:
            log("Ignored unknown command %s from" % command, user)
            text = text[:position] + text[position + 2:]

        # and this means we got the beginning of an IAC
        else:
            position += 1

    # replace the input with our cleaned-up text
    user.partial_input = text
