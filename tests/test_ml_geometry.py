import numpy as np
from src.ml.candidates import ArcCandidateIndex
from src.ml.features import FEATURE_NAMES


def test_candidate_index_finds_segment_close_to_point():
    nodes = np.array([[0.,0.],[100.,0.],[0.,100.]])
    arcs = np.array([[0,1],[0,2]])
    index = ArcCandidateIndex(nodes, arcs)
    ids, d, frac = index.query(np.array([50.,5.]), radius_m=10)
    assert 0 in ids
    pos = list(ids).index(0)
    assert abs(d[pos]-5.0) < 1e-6
    assert abs(frac[pos]-0.5) < 1e-6


def test_feature_names_present():
    assert "distance_m" in FEATURE_NAMES
    assert "heading_difference_cos" in FEATURE_NAMES
