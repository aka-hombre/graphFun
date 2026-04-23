import networkx as nx
from os import getcwd
import matplotlib.pyplot as plt


graphs = []

"""
with open("graph_data/V6/graphs6.g6")as f:
    for line in f:
        if not line: continue
        print(line.strip)
        Gx = nx.parse_graph6(line.strip())
        graphs.append(Gx)
"""

graphs = nx.read_graph6("graph_data/V6/graphs6.g6")
print(len(graphs))
plt.figure()
nx.draw(graphs[-1])
plt.show()
