# Dataset Readme

These datasets are collected in the same format as [robomimic](https://robomimic.github.io/docs/datasets/overview.html). 
To train on this dataset (and your own datasets), please modify your config file as follows:
1. Change `dataset_path` to the path of `hdf5` file.
2. Use the following `shape_meta`.

```
shape_meta:
  action:
    shape:
    - 4
  obs:
    pos:
      shape:
      - 4
      type: low_dim
    camera0:
      shape:
      - 3
      - 120
      - 160
      type: rgb
    camera1:
      shape:
      - 3
      - 120
      - 160
      type: rgb
```
