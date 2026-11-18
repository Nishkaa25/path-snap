
import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import (
    top1_accuracy,
    macro_f1,
    logloss,
    miss_rate,
    route_continuity,
    route_set_f1,
    best_threshold
)


def test_top1_accuracy():
    df = pd.DataFrame({
        "trip_id": ["1", "1", "1", "1"],
        "obs": [0, 0, 1, 1],
        "arc_id": [10, 11, 11, 12],
        "label": [1, 0, 0, 1]
    })

    p = np.array([0.9, 0.1, 0.2, 0.8])

    assert top1_accuracy(df, p, 3) == 2 / 3


def test_macro_f1():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.2, 0.8])

    score = macro_f1(y, p)

    assert 0 <= score <= 1


def test_logloss():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.2, 0.8])

    assert logloss(y, p) > 0


def test_miss_rate():
    df = pd.DataFrame({
        "trip_id": ["1", "1", "1", "1"],
        "obs": [0, 0, 1, 1],
        "label": [1, 0, 0, 1]
    })

    assert miss_rate(df, 3) == pytest.approx(1 / 3)


def test_route_continuity():
    arcs = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [8, 9]
    ])

    predicted = [0, 0, 1, 2, 3]

    assert route_continuity(
        predicted, arcs
    ) == 2 / 3


def test_route_set_f1():
    predicted = [10, 11, 12]
    reference = np.array([10, 11, 12])

    assert route_set_f1(
        predicted, reference
    ) == 1.0


def test_best_threshold():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])

    threshold, score = best_threshold(y, p)

    assert 0.05 <= threshold <= 0.95
    assert 0 <= score <= 1
