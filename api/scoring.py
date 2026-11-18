"""ONLY place the reference route is read, and only AFTER prediction. Returns numbers only."""


def score(trip_dir, clf, hmm, arcs):
    from src.common import load_trip, project
    from src.data.labels import align_labels
    from src.eval.metrics import route_continuity, route_set_f1

    ref = project(load_trip(trip_dir, with_route=True))
    lab = align_labels(ref)                       # return_dist=False -> one arc id per GPS point
    n = len(lab)
    return dict(
        top1_classifier=float((clf == lab).sum() / n),
        top1_hmm=float((hmm == lab).sum() / n),
        continuity_classifier=route_continuity(clf, arcs),
        continuity_hmm=route_continuity(hmm, arcs),
        route_f1_classifier=route_set_f1(clf, ref.route),
        route_f1_hmm=route_set_f1(hmm, ref.route),
    )
