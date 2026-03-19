# models/utils.py
"""Shared utility functions used across models and scripts."""


def rank(values):
    """Return rank array (1-based, average ties) for Spearman correlation.

    Given a list of numeric values, returns a list of ranks where ties
    receive the average of their positions. Used for computing Spearman
    rank correlation without a scipy dependency.

    Args:
        values: List of numeric values to rank.

    Returns:
        List of floats representing ranks (1-based).
    """
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks
