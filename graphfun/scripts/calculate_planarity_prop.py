import numpy as np
from graphfun.data.data_manager import DataManager


p_path = "data/metadata/graphs_manifest.parquet"
d_path = "data/graph_data/V10/"

data = DataManager(parquet_path=p_path, data_dir=d_path).to_full_dataframe()

counts = data['planar'].value_counts()
num_planar = counts.get(1, 0)
num_not_planar = counts.get(0, 0)

total = num_planar + num_not_planar

prop_planar = num_planar / total
prop_not_planar = num_not_planar / total

string= f"Total number of graphs: {total}\n ***********\n Proportion of planar: {prop_planar} \n Number of planar: {num_planar} \n ***********\n Proportion not planar: {prop_not_planar}\n Number of not planar: {num_not_planar}"

with open('data/graph_data/planprop.txt', 'w', encoding='utf-8') as f:
    f.write(string)