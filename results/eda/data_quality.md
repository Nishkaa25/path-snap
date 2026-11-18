# Data-quality log (Member A)

Partitions covered: train (80 trips) and student_test (10 trips). Source tables: `results/eda/inventory.csv`, `flagged_trips.csv`. Evaluator-testing data was not used.

## Preprocessing applied (src/common.py, src/data/labels.py)
| Step | Detail |
|---|---|
| Invalid GPS rows | Rows with NaN or out-of-range lon/lat are dropped and counted (`dropped_gps`). Result: 0 rows dropped in all 90 trips. |
| Projection | Local equirectangular projection around each trip's centroid (metres). Valid for extents of a few km; use UTM (pyproj) for very large extents. |
| Reference labels | Each GPS point is assigned a reference arc by monotone alignment along the `.route` file (route index never goes backwards, minimises point-to-arc distance, O(T x n)). Assumption: the reference route is a good but imperfect label. |
| Leakage guard | Prediction path loads with `with_route=False`. |

## Issues, evidence and decisions
| # | Issue | Trips (partition) | Evidence | Decision |
|---|---|---|---|---|
| 1 | Most GPS points far from the reference route (stationary or drifting GPS blobs not covered by the route, seen on Folium) | 33, 54 (train) | Share of points > 50 m from the reference arc: trip 33 = 58.9 %, trip 54 = 89.4 %. Median GPS-to-reference distance 121.7 m and 139.2 m (normal 4-7 m) | Exclude from supervised training (too few valid labels after masking). Keep for diagnostic evaluation only. |
| 2 | Some points far from reference route | 43, 66, 2, 81 (train) | Share of points > 50 m: 43 = 7.1 %, 66 = 9.0 %, 2 = 12.0 %, 81 = 4.4 %. p95 / p99 (m): 43 = 70.0 / 161.3; 66 = 65.2 / 635.8; 2 = 241.3 / 479.0; 81 = 44.5 / 137.2 | Keep the trip. Mask or filter observations whose GPS-to-reference distance exceeds the threshold (proposed 50 m; to be confirmed with B). Do not drop the whole trip. |
| 3 | Very large time gaps | 90, 78 (train) | Max gap 153,064 s and 52,810 s | Keep. Use real timestamps (never assume 1 s steps) and treat large gaps as sequence breaks. |
| 4 | Moderate time gaps (> 10 s) | 4, 81, 42, 11, 6, 21, 24, 44, 69, 84 and others (train); 76 (student_test) | e.g. trip 4 has 215 gaps > 10 s; trip 76 has 24 | Keep. Features and HMM transitions must use real dt. |
| 5 | Fragmented road network | 0 (student_test), 43, 54 and others | Many connected components (e.g. 100, 421, 213); giant component share 0.87 for student trip 0 | Keep. Not a GPS fault. Candidate search must not assume one connected network. |
| 6 | Reference route not fully connected | 2, 33, 75 | route_connected 0.9962, 0.9940, 0.9813 | Keep. Note as a label limitation. |
| 7 | Timestamps | all | `time_not_increasing` = 0 in every trip | No action. |
| 8 | Isolated nodes | all | `isolated_nodes` = 0 in every trip | No action. |

## Notes for teammates
- Training labels come from `align_labels`; they are reference labels, not ground truth.
- Student_test trips are clean (median 4-7 m, p95 12-26 m), apart from the fragmented network on trip 0.
- Thresholds (50 m mask) are proposals and can change after the Folium checks.

## Figures
- report/figures/map_normal_overview.png, map_normal_zoomed.png: trip 00000001 (train), GPS on the reference route.
- report/figures/map_trip33_gps_blobs.png: trip 00000033 (train), dense GPS blobs off the route.
- Generator: src/viz/make_eda_maps.py, src/viz/folium_map.py (HTML in maps/).
