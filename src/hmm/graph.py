import numpy as np


class Graph:
    """Directed road graph in CSR form. arcs: int array (n_arcs, 2) of (from_node, to_node)."""

    def __init__(self, arcs, n_nodes):
        arcs = np.asarray(arcs)
        self.frm, self.to = arcs[:, 0], arcs[:, 1]
        order = np.argsort(self.frm, kind="stable")
        self.out_arcs = order
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(self.frm, minlength=n_nodes))])
        self._reach = {}

    def outs(self, node):
        return self.out_arcs[self.indptr[node]:self.indptr[node + 1]]

    def reach(self, arc, max_h):
        """Nodes reachable after leaving `arc`, with hop count: {node: h}.
        h=0 means the very next arc may start there."""
        key = (arc, max_h)
        if key in self._reach:
            return self._reach[key]
        seen = {int(self.to[arc]): 0}
        frontier = [int(self.to[arc])]
        for h in range(1, max_h + 1):
            nxt = []
            for node in frontier:
                for a in self.outs(node):
                    v = int(self.to[a])
                    if v not in seen:
                        seen[v] = h
                        nxt.append(v)
            frontier = nxt
        self._reach[key] = seen
        return seen
