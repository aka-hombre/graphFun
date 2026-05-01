<h1 style="display: flex; align-items: center; gap: 10px;">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20221103114655/PolyhedronGraph-660x472.png" alt="alt text" width="70">
  <span>graphFun</span>
</h1>

*machine learning & graph playground*
___

Non-isomorphic graphs found in `data/graph_data/` were generated using `nauty` using the tool
`geng`

```
./geng -q 10 | split -l 107189 -d -a 3 --additional-suffix=.g6 - batches/graphs10_
```

Each file has 107189, and there are 112 files totaling 12005168 non-isomorphic graphs.

___

Running `python graphfun/data/build_manifest.py` generates a `.parquet` file to train on in `data/metatdata` this is saved locally and ignored by `.gitignore`, because it exceeds the filesize limits on github. 

It is reccomended to run the script on HPC, then transfer locally.

___
To run
1. `apptainer build mySif.sif containers/apptainer.def`
2. `apptainer exec mySif.sif python3 graphfun/scripts/train_linear.py`
