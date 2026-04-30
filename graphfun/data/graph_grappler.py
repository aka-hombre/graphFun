import networkx as nx 
import pandas as pd
from typing import List, Tuple, Union
from numpy.typing import NDArray
import numpy as np

def get_graphs(df: pd.DataFrame, 
               return_adj_flat: bool=False
               )-> Tuple[Union[List[nx.Graph], NDArray], NDArray]:
    """
    Takes a DataFrame processed by data_manager, and extracts the graph6
    representations along with the planar label
    
    Args:
        df: pandas DataFrame processed by data_manager
        return_adj_flat: option to return as a networkx Graph object or numpy adjacency matrix (FLATTENED)
    Returns:
        Tuple[Union[List[nx.Graph], NDArray], NDArray]:
            graphs:
                List of networkx.Graph objects or NumPy array of adjacency matrices, as (1x100) vec,
                depending on `return_adj`
            labels:
                NumPy array of boolean planarity labels
    """
    graphs = df['graph6_rep'].to_list()
    
    #labels = np.array(df['planar'].to_list(), dtype=int)
    labels = np.array(df['planar'].to_list())
    #G = [nx.from_graph6_bytes(g.strip().encode()) for g in graphs]
    

    if return_adj_flat: 
        features = np.stack([
            nx.to_numpy_array(
                nx.from_graph6_bytes(g.encode())
            ).flatten().astype(np.float32)
            for g in graphs                     # iterate strings, not Graph objects, flattens for input to linear as is, avoide bottlenecking when loading in the whole dataset
        ])
        return features, labels
    else: 
        G = [nx.from_graph6_bytes(g.encode()) for g in graphs]
        return G, labels

