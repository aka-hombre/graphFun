#!/usr/bin/env python

from pathlib import Path
import re

import networkx as nx
import pandas as pd


BATCH_RE = re.compile(r"graphs10_(\d{3})\.g6$")


def parse_batch_id(path: Path) -> int:
    m = BATCH_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return int(m.group(1))


def build_manifest(input_dir: str, output_path: str):
    input_dir = Path(input_dir)
    records = []
    global_idx = 0

    for path in sorted(input_dir.glob("graphs10_*.g6")):
        batch_id = parse_batch_id(path)
        graphs = nx.read_graph6(path)

        for local_idx, g in enumerate(graphs):
            print(f"processing batch {batch_id}")
            records.append(
                {
                    "sample_id": f"{batch_id:03d}_{local_idx:09d}",
                    "batch_id": batch_id,
                    "source_file": path.name,
                    "local_idx": local_idx,
                    "global_idx": global_idx,
                    "num_nodes": 10,
                    "num_edges": g.number_of_edges(),
                    "planar": nx.is_planar(g),
                }
            )
            global_idx += 1

    df = pd.DataFrame(records)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


if __name__ == "__main__":
    build_manifest("graph_data/V10/", "data/manifests/graphs_manifest.parquet")