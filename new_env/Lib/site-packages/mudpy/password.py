"""Password hashing functions and constants for the mudpy engine."""

# Copyright (c) 2004-2019 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

import passlib.context

_CONTEXT = passlib.context.CryptContext(
    default="pbkdf2_sha512", pbkdf2_sha512__default_rounds=1000,
    schemes=["pbkdf2_sha512"])


def create(password):
    return _CONTEXT.hash(password)


def verify(password, encoded_hash):
    return _CONTEXT.verify(password, encoded_hash)
