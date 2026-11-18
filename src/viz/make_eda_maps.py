from src.common import load_trip, project
from src.viz.folium_map import make_map

for part, tid, name in [("train", "00000001", "eda_normal"),
                        ("train", "00000033", "problem_33"),
                        ("train", "00000043", "problem_43"),
                        ("train", "00000054", "problem_54")]:
    make_map(project(load_trip(f"data/{part}/{tid}")), out_html=f"maps/{name}.html")
    print("saved", name)
