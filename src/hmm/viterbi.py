import numpy as np
from dataclasses import dataclass


@dataclass
class HMMConfig:
    lam: float = 1.0        # emission weight
    alpha: float = 1.0      # cost per hop
    uturn: float = 3.0      # extra penalty for reversing onto the twin arc
    log_eps: float = -12.0  # unreachable transition
    base_hops: int = 3
    extra_hops_per_s: float = 1.0
    max_hops_cap: int = 12
    max_gap: float = 30.0   # seconds; larger gap restarts the chain


def trans_matrix(prev_arcs, arcs, g, cfg, gap):
    """log A(i, j) for every (previous candidate i, current candidate j)."""
    max_h = min(cfg.max_hops_cap, cfg.base_hops + int(max(gap - 1, 0) * cfg.extra_hops_per_s))
    M = np.full((len(prev_arcs), len(arcs)), cfg.log_eps)
    for a, i in enumerate(prev_arcs):
        r = g.reach(int(i), max_h)
        for b, j in enumerate(arcs):
            if i == j:
                M[a, b] = 0.0
                continue
            h = r.get(int(g.frm[j]))
            if h is not None:
                M[a, b] = -cfg.alpha * (h + 1)
                if g.frm[j] == g.to[i] and g.to[j] == g.frm[i]:
                    M[a, b] -= cfg.uturn
    return M


def backtrack(seg, delta, pred):
    k = int(np.argmax(delta))
    for t, arcs, psi in reversed(seg):
        pred[t] = arcs[k]
        if psi is not None:
            k = psi[k]


def decode(arcs_per_obs, p_per_obs, times, g, cfg):
    """arcs_per_obs[t]: candidate arc ids (or None/empty); p_per_obs[t]: classifier probabilities.
    Returns predicted arc id per observation (-1 where no candidates)."""
    T = len(times)
    pred = np.full(T, -1, dtype=int)
    seg, delta, prev_arcs, prev_t = [], None, None, None
    for t in range(T):
        arcs = arcs_per_obs[t]
        if arcs is None or len(arcs) == 0:
            continue                                   # missing candidates: bridge over
        lb = cfg.lam * np.log(np.clip(p_per_obs[t], 1e-6, 1.0))
        gap = (times[t] - prev_t) if prev_t is not None else 0.0
        if delta is None or gap > cfg.max_gap:
            if delta is not None:
                backtrack(seg, delta, pred)
            delta, seg = lb, [(t, arcs, None)]         # (re)start a segment
        else:
            S = delta[:, None] + trans_matrix(prev_arcs, arcs, g, cfg, gap)
            psi = S.argmax(0)
            delta = S.max(0) + lb
            delta = delta - delta.max()                # numeric stability, argmax unchanged
            seg.append((t, arcs, psi))
        prev_arcs, prev_t = arcs, times[t]
    if delta is not None:
        backtrack(seg, delta, pred)
    return pred
