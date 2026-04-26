# :crystal_ball: **graphFun**
## *machine learning & graph playground*

Non-isomorphic graphs found in `data/graph_data/` were generated using `nauty` using the tool
`geng`

```
./geng -q 10 | split -l 107189 -d -a 3 --additional-suffix=.g6 - batches/graphs10_
```

Each file has 107189, and there are 111 files totaling 1320675669 non-isomorphic graphs.

___

Running `python graphfun/data/build_manifest.py` generates a `.parquet` file to train on in `data/metatdata` this is saved locally and ignored by `.gitignore`, because it exceeds the filesize limits on github. 

It is reccomended to run the script on HPC, then transfer locally.