import networkx as nx
from os import getcwd
from pathlib import Path
from graphfun.data.data_manager import DataManager
from graphfun.data.graph_grappler import get_graphs

import pandas as pd 


graphs = []

"""
with open("graph_data/V6/graphs6.g6")as f:
    for line in f:
        if not line: continue
        print(line.strip)
        Gx = nx.parse_graph6(line.strip())
        graphs.append(Gx)


graphs = nx.read_graph6("graph_data/V6/graphs6.g6")
print(len(graphs))
plt.figure()
nx.draw(graphs[-1])
plt.show()

"""
p_path = "data/metadata/graphs_manifest.parquet"
d_path = "data/graph_data/V10/"
df = pd.read_parquet("data/metadata/graphs_manifest.parquet")

shard1 = DataManager(p_path, d_path).pull_shard(000)

#print(shard1.head())

#shard1 = DataManager(p_path, d_path).pull_slice(0,3)

#print(shard1.head())

G, labels = get_graphs(shard1, return_adj=True)
print(type(G))