import numpy as np
from src.hmm.graph import Graph
from src.hmm.viterbi import HMMConfig, decode, trans_matrix


def make_graph():
    # nodes 0..5; arcs (id: from->to)
    # 0: 0->1   1: 0->2        (step 0 candidates)
    # 2: 1->3   3: 4->5        (step 1: arc 2 is connected to arc 0, arc 3 is disconnected)
    # 4: 3->4                  (step 2 candidate, continues from arc 2 via node 3->4)
    # 5: 5->0                  (step 2 alternative, continues from arc 3)
    arcs = np.array([[0, 1], [0, 2], [1, 3], [4, 5], [3, 4], [5, 0]])
    return Graph(arcs, 6)


def test_viterbi_prefers_connected_path():
    g = make_graph()
    cfg = HMMConfig()
    arcs = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    # classifier slightly prefers the DISCONNECTED candidates (arc 3 at step 1, arc 5 at step 2)
    p = [np.array([0.8, 0.2]), np.array([0.45, 0.55]), np.array([0.45, 0.55])]
    times = np.array([0.0, 1.0, 2.0])

    clf_only = [int(a[np.argmax(q)]) for a, q in zip(arcs, p)]
    assert clf_only == [0, 3, 5]                       # classifier alone teleports

    pred = decode(arcs, p, times, g, cfg)
    assert list(pred) == [0, 2, 4]                     # Viterbi picks the connected route


def test_missing_candidates_are_bridged():
    g = make_graph()
    cfg = HMMConfig()
    arcs = [np.array([0, 1]), None, np.array([2, 3])]
    p = [np.array([0.8, 0.2]), None, np.array([0.45, 0.55])]
    pred = decode(arcs, p, np.array([0.0, 1.0, 2.0]), g, cfg)
    assert pred[1] == -1 and pred[0] == 0 and pred[2] == 2


def test_gap_restarts_chain():
    g = make_graph()
    cfg = HMMConfig(max_gap=30.0)
    arcs = [np.array([0, 1]), np.array([2, 3])]
    p = [np.array([0.8, 0.2]), np.array([0.1, 0.9])]
    # gap of 100 s > max_gap: no transition penalty applies, so the classifier's pick (arc 3) stands
    pred = decode(arcs, p, np.array([0.0, 100.0]), g, cfg)
    assert list(pred) == [0, 3]


def test_uturn_penalty():
    arcs = np.array([[0, 1], [1, 0], [1, 2]])
    g = Graph(arcs, 3)
    cfg = HMMConfig(alpha=1.0, uturn=3.0)
    M = trans_matrix(np.array([0]), np.array([1, 2]), g, cfg, gap=1.0)
    assert M[0, 0] == -1.0 - 3.0                       # reverse onto the twin arc: hop cost + U-turn
    assert M[0, 1] == -1.0                             # forward continuation: hop cost only
