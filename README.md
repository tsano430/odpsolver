odpsolver
---------

The order/degree problem is the problem of finding a graph that has smallest diameter and Average Shortest Path Length (ASPL) for a given order and degree. 
odpsolver is a genetic algorithm [1] for tackling the problem. 

Usage
-----

Given the order 100 and degree 3, run the following command: 
```
$ Python odpsolver.py 100 3
Diameter: 
ASPL: 
```
As a result, `regular-graph-o100-d3.dat` is obtained.

Dependency
----------

- numpy
- scipy
- networkx
- joblib
- optuna


Reference
---------

[1] Reiji Hayashi, Tsuyoshi Migita and Norikazu Takahashi, ``A genetic algorithm for finding regular graphs with minimum average shortest path length,'' Proceedings of the 2020 IEEE Symposium Series on Computational Intelligence (SSCI2020), pp.-, 2020.