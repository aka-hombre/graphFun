import networkx as nx 
import pandas as pd
from typing import List, Tuple
import numpy as np

def get_graphs(df: pd.DataFrame, 
               return_adj: bool=False, 
               bool_as_int: bool =False
               )-> Tuple[List, List]:
    """
    Takes a DataFrame processed by data_manager, and extracts the graph6
    representations along with the planar label
    
    Args:
        df: pandas DataFrame processed by data_manager
        return_adj: option to return as a networkx Graph object or numpy adjacency matrix
        bool_as_int: option to return booleans as integers
    Returns:
        Tuple[List, List]: List of Graphs (object specified by argument) and List of boolean planarity labels
    """
    graphs = df['graph6_rep'].to_list()

    if not bool_as_int: labels = df['planar'].to_list()

    else: labels = list(map(int, df['planar'].to_list()))

    G = [nx.from_graph6_bytes(g.strip().encode()) for g in graphs]  

    if not return_adj: return G, labels
    else: return graphList_to_adj(G), labels

def graphList_to_adj(G:List[nx.Graph]):
    """
    Takes a networkx Graph and returns an adjacency matrix
    """
    return [nx.to_numpy_array(g, dtype=np.ndarray) for g in G]