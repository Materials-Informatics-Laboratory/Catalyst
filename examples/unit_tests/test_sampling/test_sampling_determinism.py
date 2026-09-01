"""Determinism checks for sampling routines used by the public examples."""

from __future__ import annotations

import numpy as np

from catalyst.utilities.sampling import run_sampling


def test_random_sampling_is_reproducible_for_a_fixed_generator_seed():
    data = np.arange(60, dtype=float).reshape(20, 3)
    first = run_sampling(
        data,
        sampling_type="random",
        split=0.35,
        rng=np.random.default_rng(112358),
        params_group={"clusters": 1},
    )
    second = run_sampling(
        data,
        sampling_type="random",
        split=0.35,
        rng=np.random.default_rng(112358),
        params_group={"clusters": 1},
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_kmeans_sampling_is_reproducible_for_a_fixed_generator_seed():
    rng = np.random.default_rng(7)
    data = np.vstack(
        (
            rng.normal(loc=-2.0, scale=0.1, size=(12, 2)),
            rng.normal(loc=0.0, scale=0.1, size=(12, 2)),
            rng.normal(loc=2.0, scale=0.1, size=(12, 2)),
        )
    )
    first = run_sampling(
        data,
        sampling_type="kmeans",
        split=0.5,
        rng=np.random.default_rng(42),
        params_group={"clusters": 3},
    )
    second = run_sampling(
        data,
        sampling_type="kmeans",
        split=0.5,
        rng=np.random.default_rng(42),
        params_group={"clusters": 3},
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
