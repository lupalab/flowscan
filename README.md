# FlowScans
### A likelihood estimator for exchangeable data

This is the official repository for the [2020 AAAI article](http://arxiv.org/abs/1902.01967).

### General Idea
We utilize (learned) permutaton equivariant flows to transform the data to a space that is better suited to scanning. We then scan over each the first dimension of each point to convert the set into a sequence. Finally, we apply conventional likelihood estimators (such as autoregressive flows).

![Permutation Equivariant Flows over a plane set][flow]

## Results
We compare to several other general, exchangeable likelihood estimators, [BRUNO](https://arxiv.org/abs/1802.07535) and [The Neural Statistician](https://arxiv.org/abs/1606.02185).

| DataSet      | BRUNO | NS    | FlowScan
| :----------- | :---: | :---: | :------: |
| Synthetic    | -2.28 | -1.07 | **0.14**
| Airplanes    |  2.71 |  4.09 | **4.81**
| Chairs       |  0.75 |  2.02 | **2.58**
| ModelNet10   |  0.49 |  2.12 | **3.01**
| ModelNet10a  |  1.20 |  2.82 | **3.58**
| Caudate      |  1.29 |  4.49 | **4.87**
| Thalamus     |  0.82 |  2.69 | **3.12**
| SpatialMNIST | -5.68 | -5.37 | **-5.26**




## Use
To use this version of the code you can build the provided Dockerfile or can install the necessary packages.
`pip install dill tqdm requests matplotlib scipy tensorflow-gpu==1.12.0` should suffice.

To train a model, the data must be provided in a pickle file.
On load, the data must be returned as a dictionary of arrays with keys `train`, `valid`, and `test`.
Each of these (3D) arrays must be organized as sets, points, and dimensions such that indexing into the first dimension (`data['train'][n])`) will return a complete set.

The default model is trained by calling
```python
import flowscan.demos.flowscan as fdemo
dataset = 'plane'               # plane sets from ModelNet10
datadir = '/home/me/data/sets'  # the directory in which the data is stored
results = fdemo.main(
    dataset=dataset, datadir=datadir, dims=3, subsample=512, train_iters=40000)
```


## Citation
```
@inproceedings{bender2020flowscan,
  author    = {Christopher M.~Bender and
               Kevin O'Connor and
               Yang Li and
               Juan Jose Garcia and
               Manzil Zaheer and
               Junier B.~Oliva},
  title     = {Exchangeable Generative Models with Flow Scans},
  publisher = {{AAAI} Press},
  url       = {http://arxiv.org/abs/1902.01967},
  year      = {2020},
}
```


[flow]: https://github.com/lupalab/flowscan_beta/blob/master/images/planeflow.png?raw=true "Plane PEq Flow"
[fs_mnist]: https://github.com/lupalab/flowscan_beta/blob/master/images/fs_mnist.png?raw=true "FS MNIST Samples"
