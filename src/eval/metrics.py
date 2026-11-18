
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, log_loss


def top1_accuracy(df, p, n_obs_total):
    """
    Observation-level Top-1 Accuracy.

    For every GPS observation, select the candidate road
    with the highest predicted probability.

    Denominator = ALL GPS observations.
    Observations without candidates count as incorrect.
    """

    if n_obs_total <= 0:
        raise ValueError("n_obs_total must be positive")

    if len(df) != len(p):
        raise ValueError("Dataframe and probability lengths differ")

    if df.empty:
        return 0.0

    d = pd.DataFrame({
        "trip_id": df["trip_id"].values,
        "obs": df["obs"].values,
        "label": df["label"].values,
        "p": np.asarray(p)
    })

    best_idx = d.groupby(
        ["trip_id", "obs"]
    )["p"].idxmax()

    best = d.loc[best_idx]

    return float(best["label"].sum() / n_obs_total)


def macro_f1(y, p, thr=0.5):
    """
    Binary candidate classification Macro F1.

    Class 1 = candidate is the reference arc.
    Class 0 = candidate is not the reference arc.

    Macro F1 gives equal importance to both classes.
    """

    predictions = (np.asarray(p) >= thr).astype(int)

    return float(f1_score(
        y,
        predictions,
        labels=[0, 1],
        average="macro",
        zero_division=0
    ))


def logloss(y, p):
    """
    Calculate binary Log Loss.

    Lower Log Loss indicates better probability estimates.
    Probabilities are clipped to prevent log(0).
    """

    p = np.clip(np.asarray(p), 1e-7, 1 - 1e-7)

    return float(log_loss(
        y,
        p,
        labels=[0, 1]
    ))


def miss_rate(df, n_obs_total):
    """
    Candidate Miss Rate.

    Measures the proportion of GPS observations
    whose reference arc is absent from the candidate set.
    """

    if n_obs_total <= 0:
        raise ValueError("n_obs_total must be positive")

    if df.empty:
        return 1.0

    hit = df.groupby(
        ["trip_id", "obs"]
    )["label"].max().sum()

    return float(1 - hit / n_obs_total)


def route_continuity(pred_arcs, arcs):
    """
    Measures the proportion of consecutive DISTINCT
    predicted road arcs that are connected.

    arcs[:, 0] = from_node
    arcs[:, 1] = to_node

    Arc IDs must correspond to row indices in this
    trip's original .arcs file.
    """

    sequence = [
        a for a in pred_arcs if a >= 0
    ]

    sequence = [
        a for i, a in enumerate(sequence)
        if i == 0 or a != sequence[i - 1]
    ]

    if len(sequence) < 2:
        return 1.0

    s = np.asarray(sequence, dtype=int)
    arcs = np.asarray(arcs)

    if s.min() < 0 or s.max() >= len(arcs):
        raise ValueError("Predicted arc ID outside arc array")

    connected = (
        arcs[s[:-1], 1] == arcs[s[1:], 0]
    )

    return float(np.mean(connected))


def route_set_f1(pred_arcs, route):
    """
    Sequence-level set-based F1.

    Compares the distinct predicted arc IDs against
    the distinct reference route arc IDs.

    This measures arc coverage, not chronological order.
    """

    predicted = set(
        a for a in pred_arcs if a >= 0
    )

    reference = set(
        np.asarray(route).ravel()
    )

    tp = len(predicted & reference)

    if tp == 0:
        return 0.0

    precision = tp / len(predicted)
    recall = tp / len(reference)

    return float(
        2 * precision * recall / (precision + recall)
    )


def best_threshold(y, oof_probabilities):
    """
    Select the probability threshold using training
    out-of-fold predictions ONLY.

    Never select the threshold using student-test
    or evaluator-test results.
    """

    thresholds = np.arange(0.05, 0.951, 0.01)

    scores = [
        macro_f1(y, oof_probabilities, thr)
        for thr in thresholds
    ]

    best_score = max(scores)

    # Deterministic tie-breaking: prefer threshold
    # closest to 0.5.
    best_candidates = [
        t for t, score in zip(thresholds, scores)
        if np.isclose(score, best_score)
    ]

    best_thr = min(
        best_candidates,
        key=lambda t: abs(t - 0.5)
    )

    return float(best_thr), float(best_score)
