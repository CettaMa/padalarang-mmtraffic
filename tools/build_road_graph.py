#!/usr/bin/env python3
"""Build road network graph and adjacency files for the 12 ROI nodes."""

import json
import math
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import canonical node ordering
import sys
sys.path.insert(0, str(REPO_ROOT / "live"))
from forecast.common import NODES


def haversine(p1, p2):
    R = 6371000.0  # Earth radius in meters
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def min_endpoint_dist(path1, path2):
    ends1 = [path1[0], path1[-1]]
    ends2 = [path2[0], path2[-1]]
    return min(haversine(p, q) for p in ends1 for q in ends2)


def main():
    graph_dir = REPO_ROOT / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load segments and ROI definitions
    seg_file = REPO_ROOT / "live" / "segments.json"
    seg_doc = json.loads(seg_file.read_text(encoding="utf-8"))
    source_str = seg_doc.get("_source", "")
    seg_map = {(s["cctv"], s["roi_idx"]): s for s in seg_doc["segments"]}

    # Load ROI lengths
    roi_lengths = {}
    for p in (REPO_ROOT / "rois").glob("*_roi.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        cctv_id = d["cctv_id"]
        for idx, r in enumerate(d["rois"]):
            roi_lengths[(cctv_id, idx)] = float(r["length_meters"])

    N = len(NODES)

    # 2. Build nodes.csv
    nodes_rows = []
    for i, node_key in enumerate(NODES):
        cctv_name, roi_str = node_key.split("|")
        roi_idx = int(roi_str)
        seg = seg_map[(cctv_name, roi_idx)]
        path = seg["path"]
        length_m = roi_lengths.get((cctv_name, roi_idx), 0.0)
        nodes_rows.append({
            "node_id": i,
            "node_key": node_key,
            "cctv_name": cctv_name,
            "roi_idx": roi_idx,
            "roi_label": f"ROI {roi_idx + 1}",
            "start_lat": path[0][0],
            "start_lon": path[0][1],
            "end_lat": path[-1][0],
            "end_lon": path[-1][1],
            "length_m": length_m,
        })
    df_nodes = pd.DataFrame(nodes_rows)
    df_nodes.to_csv(graph_dir / "nodes.csv", index=False)
    print(f"[✓] Saved {graph_dir / 'nodes.csv'} ({len(df_nodes)} nodes)")

    # 3. Compute distance matrix and edge connections
    dist_mx = [[0.0] * N for _ in range(N)]
    adj_mx = [[0] * N for _ in range(N)]
    edge_rows = []

    for i in range(N):
        cctv_i, roi_i = NODES[i].split("|")
        path_i = seg_map[(cctv_i, int(roi_i))]["path"]
        for j in range(i + 1, N):
            cctv_j, roi_j = NODES[j].split("|")
            path_j = seg_map[(cctv_j, int(roi_j))]["path"]
            d = round(min_endpoint_dist(path_i, path_j), 1)
            dist_mx[i][j] = d
            dist_mx[j][i] = d

            # Connection rule: same camera node OR minimum endpoint distance <= 120.0 meters
            if cctv_i == cctv_j or d <= 120.0:
                adj_mx[i][j] = 1
                adj_mx[j][i] = 1
                edge_rows.append({
                    "src": i,
                    "dst": j,
                    "src_key": NODES[i],
                    "dst_key": NODES[j],
                    "distance_m": d,
                    "type": "intra_camera" if cctv_i == cctv_j else "cross_camera",
                })

    assert len(edge_rows) == 18, f"Expected 18 edges, got {len(edge_rows)}"

    # 4. Save distance_m.csv, edges.csv, adjacency.csv
    df_dist = pd.DataFrame(dist_mx, index=NODES, columns=NODES)
    df_dist.to_csv(graph_dir / "distance_m.csv")
    print(f"[✓] Saved {graph_dir / 'distance_m.csv'}")

    df_edges = pd.DataFrame(edge_rows)
    df_edges.to_csv(graph_dir / "edges.csv", index=False)
    print(f"[✓] Saved {graph_dir / 'edges.csv'} ({len(df_edges)} edges)")

    df_adj = pd.DataFrame(adj_mx, index=NODES, columns=NODES)
    df_adj.to_csv(graph_dir / "adjacency.csv")
    print(f"[✓] Saved {graph_dir / 'adjacency.csv'}")

    # 5. README.txt
    readme_text = (
        "Road Network Topology for Padalarang Corridor Traffic Dataset (v1.0.0)\n"
        "====================================================================\n\n"
        f"Source attribution:\n{source_str}\n\n"
        "Contents:\n"
        "- nodes.csv: List of 12 observation nodes (ROI polygons) across 6 CCTV cameras with coordinates and lengths.\n"
        "- edges.csv: 18 undirected edges linking adjacent/connected road segments.\n"
        "  * Rule: An edge connects two ROIs if they share a camera installation (intra-camera opposite/turning lanes)\n"
        "    or if their traced segment endpoints lie within 120 meters (cross-camera proximity on the road network).\n"
        "- adjacency.csv: 12x12 binary, unweighted symmetric adjacency matrix A in {0, 1}^(12x12) per Eq. (15).\n"
        "- distance_m.csv: 12x12 pairwise minimum endpoint distance matrix in meters.\n"
    )
    (graph_dir / "README.txt").write_text(readme_text, encoding="utf-8")
    print(f"[✓] Saved {graph_dir / 'README.txt'}")


if __name__ == "__main__":
    main()
