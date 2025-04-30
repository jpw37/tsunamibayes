"""Utility functions for menu operation."""

# Copyright (c) 2019 mudpy authors. Permission to use, copy,
# modify, and distribute this software is granted under terms
# provided in the LICENSE file distributed with this software.

import mudpy


def activate_avatar_action(user):
    """Activate the selected avatar in the activate_avatar state."""
    return user.activate_avatar_by_index(int(user.choice) - 1)


def activate_avatar_action_a(user):
    """Abort selection by doing nothing."""
    return True


def activate_avatar_create(user):
    """List available avatars as choices for the activate_avatar state."""
    return dict(
        [(str(x + 1), y) for x, y in enumerate(user.list_avatar_names())])


def checking_new_account_name_action_d(user):
    """Destroy the new account because the user asked to disconnect."""
    return user.account.destroy()


def checking_new_account_name_action_g(user):
    """Destroy the new account, the user asked to go back to name entry."""
    return user.account.destroy()


def choose_gender_action(user):
    """Set the avatar's gender to the selected value."""
    return user.avatar.set("gender", user.menu_choices[user.choice])


def choose_name_action(user):
    """Set the avatar's name to the selected value."""
    return user.avatar.set("name", user.menu_choices[user.choice])


def choose_name_create_1(user):
    """Provide a randomly-generated name as choice 1."""
    return mudpy.misc.random_name()


def choose_name_create_2(user):
    """Provide a randomly-generated name as choice 2."""
    return mudpy.misc.random_name()


def choose_name_create_3(user):
    """Provide a randomly-generated name as choice 3."""
    return mudpy.misc.random_name()


def choose_name_create_4(user):
    """Provide a randomly-generated name as choice 4."""
    return mudpy.misc.random_name()


def choose_name_create_5(user):
    """Provide a randomly-generated name as choice 5."""
    return mudpy.misc.random_name()


def choose_name_create_6(user):
    """Provide a randomly-generated name as choice 6."""
    return mudpy.misc.random_name()


def choose_name_create_7(user):
    """Provide a randomly-generated name as choice 7."""
    return mudpy.misc.random_name()


def delete_account_action_y(user):
    """Permanently delete the account and all avatars, as requested."""
    return user.destroy()


def delete_avatar_action(user):
    """Delete the selected avatar."""
    return user.delete_avatar(
        user.account.get("avatars")[int(user.choice) - 1])


def delete_avatar_action_a(user):
    """Abort avatar deletion."""
    return True


def delete_avatar_create(user):
    """List available avatars for possible deletion."""
    return dict(
        [(str(x + 1), y) for x, y in enumerate(user.list_avatar_names())])


def main_utility_action_c(user):
    """Create a new avatar."""
    return user.new_avatar()


def main_utility_demand_a(user):
    """Only include avatar activation if the account has avatars."""
    return user.account.get("avatars")


def main_utility_demand_c(user):
    """Only include avatar creation if avatar count is below the limit."""
    return (len(user.account.get("avatars")) <
            mudpy.misc.universe.contents["mudpy.limit"].get("avatars"))


def main_utility_demand_d(user):
    """Only include avatar deletion if the account has avatars."""
    return user.account.get("avatars")
