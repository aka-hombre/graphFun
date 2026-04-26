import networkx as nx 
import pandas as pd
from typing import List, Tuple, Union
from numpy.typing import NDArray
import numpy as np

def get_graphs(df: pd.DataFrame, 
               return_adj: bool=False
               )-> Tuple[Union[List[nx.Graph], NDArray], NDArray]:
    """
    Takes a DataFrame processed by data_manager, and extracts the graph6
    representations along with the planar label
    
    Args:
        df: pandas DataFrame processed by data_manager
        return_adj: option to return as a networkx Graph object or numpy adjacency matrix
    Returns:
        Tuple[Union[List[nx.Graph], NDArray], NDArray]:
            graphs:
                List of networkx.Graph objects or NumPy array of adjacency matrices,
                depending on `return_adj`
            labels:
                NumPy array of boolean planarity labels
    """
    graphs = df['graph6_rep'].to_list()
    
    labels = np.array(df['planar'].to_list(), dtype=int)

    G = [nx.from_graph6_bytes(g.strip().encode()) for g in graphs]

    if return_adj: return np.array(graphList_to_adj(G)), labels
    else: return G, labels

def graphList_to_adj(G:List[nx.Graph]):
    """
    Takes a networkx Graph and returns an adjacency matrix
    """
    return [nx.to_numpy_array(g, dtype=np.ndarray) for g in G]