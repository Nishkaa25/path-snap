"""Folium map: GPS, road network, reference route, predicted route, legend, layer control.
Usage (example):
    from src.common import load_trip, project
    from src.viz.folium_map import make_map
    trip = project(load_trip("data/train/00000033"))
    make_map(trip, out_html="maps/trip_00000033.html")
NOTE: the evaluator UI must NOT show the reference route (show_reference=False there)."""
from __future__ import annotations
import numpy as np
import folium


def seg_coords(trip, arc_ids):
    a = trip.arcs[np.asarray(arc_ids, dtype=int)]
    return [[[trip.nodes[i, 1], trip.nodes[i, 0]], [trip.nodes[j, 1], trip.nodes[j, 0]]] for i, j in a]


def make_map(trip, pred_arcs=None, show_reference=True, out_html="maps/trip.html",
             max_net_arcs=8000, pad_deg=0.003):
    m = folium.Map(location=[float(trip.lat.mean()), float(trip.lon.mean())], zoom_start=15, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", attr="Esri")

    # only the road network near the track (full networks can have 300k arcs)
    lon_ok = (trip.nodes[:, 0] > trip.lon.min() - pad_deg) & (trip.nodes[:, 0] < trip.lon.max() + pad_deg)
    lat_ok = (trip.nodes[:, 1] > trip.lat.min() - pad_deg) & (trip.nodes[:, 1] < trip.lat.max() + pad_deg)
    near = lon_ok & lat_ok
    sel = np.where(near[trip.arcs[:, 0]] & near[trip.arcs[:, 1]])[0][:max_net_arcs]
    g = folium.FeatureGroup("Road network", show=True)
    if len(sel):
        folium.PolyLine(seg_coords(trip, sel), color="#999999", weight=1).add_to(g)
    g.add_to(m)

    g = folium.FeatureGroup("GPS observations")
    stride = max(1, len(trip.lat) // 1500)
    for la, lo in zip(trip.lat[::stride], trip.lon[::stride]):
        folium.CircleMarker([float(la), float(lo)], radius=2, color="#1f77b4", fill=True).add_to(g)
    g.add_to(m)

    if show_reference and trip.route is not None:
        g = folium.FeatureGroup("Reference route")
        folium.PolyLine(seg_coords(trip, trip.route), color="#2ca02c", weight=5, opacity=0.8).add_to(g)
        g.add_to(m)
    if pred_arcs is not None:
        p = [a for a in pred_arcs if a >= 0]
        g = folium.FeatureGroup("Predicted route")
        if p:
            folium.PolyLine(seg_coords(trip, p), color="#d62728", weight=3, opacity=0.9).add_to(g)
        g.add_to(m)

    legend = ("<div style='position:fixed;bottom:30px;left:30px;z-index:9999;background:white;"
              "padding:8px;border:1px solid #888;font-size:13px'><b>Legend</b><br>"
              "<span style='color:#1f77b4'>&#9679;</span> GPS observations<br>"
              "<span style='color:#2ca02c'>&#9644;</span> Reference route<br>"
              "<span style='color:#d62728'>&#9644;</span> Predicted route<br>"
              "<span style='color:#999'>&#9644;</span> Road network</div>")
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)
    m.save(out_html)
    return m
