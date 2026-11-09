"""Statistics for pass rates on small task sets: intervals and paired tests.

Pass rates on a few dozen tasks are noisy, so differences are reported with intervals and a
paired test (the same tasks in both runs), not as bare deltas.
"""
from __future__ import annotations

import math
import random

Z_95 = 1.96
BOOTSTRAP_RESAMPLES = 4000


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well-behaved for small n and extremes)."""
    if trials == 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def mcnemar_exact_p_value(only_a_passed: int, only_b_passed: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant-pair counts."""
    discordant = only_a_passed + only_b_passed
    if discordant == 0:
        return 1.0
    smaller = min(only_a_passed, only_b_passed)
    one_tail = sum(math.comb(discordant, count) for count in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * one_tail)


def paired_bootstrap_interval(differences: list[int], resamples: int = BOOTSTRAP_RESAMPLES,
                              seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval for the mean of paired differences (each in {-1, 0, 1})."""
    if not differences:
        return (0.0, 0.0)
    generator = random.Random(seed)
    count = len(differences)
    means = sorted(sum(generator.choice(differences) for _ in range(count)) / count for _ in range(resamples))
    return (means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1])
