"""Indexing helpers for the one-hot layout."""


def qubit_index(user: int, channel: int, n_channels: int) -> int:
    return user * n_channels + channel

