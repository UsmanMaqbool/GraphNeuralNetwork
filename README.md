#GraphNeuralNetwork
"In-depth and simple graph neural network: GNN principle analysis" supporting code

### About Errata

>Due to the limited level of the author and the rush of time, there will inevitably be some errors or inaccuracies in the book, which have caused trouble to readers and friends. I apologize.
The [errata] (./errata.pdf) of some problems that have been found so far is provided in the warehouse, and I would like to express my thanks to the readers who corrected these errors.

* In the introduction of the graph filter in Section 5.4, there are some description errors and vague concepts, which may cause deviations in the reader's understanding. The errata corrects the relevant problems

### Environment dependencies
````
python>=3.6
jupyter
scipy
numpy
matplotlib
torch>=1.2.0
````

### Getting Started

* [x] [Chapter5: GCN-based node classification](./chapter5)
* [x] [Chapter7: GraphSage example](./chapter7)
* [x] [Chapter8: Example of graph classification](./chapter8)
* [x] [Chapter9: Graph Autoencoder](./chapter9)

### FAQ

1. Cora dataset cannot be downloaded

The address of the Cora dataset is: [kimiyoung/planetoid](https://github.com/kimiyoung/planetoid/tree/master/data).
~~The repository provides a copy of the cora data used, which can be placed in the `chapter5/cora/raw` or `chapter7/cora/raw` directory respectively. ~~
The new code uses local data directly.