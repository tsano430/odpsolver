odpsolver
---------

The [order/degree problem][problem] is the problem of finding a graph that has smallest diameter and Average Shortest Path Length (ASPL) for a given order and degree. 
odpsolver is a genetic algorithm [1] for tackling the problem. 

Usage
-----

Given the order 16 and degree 4, run the following command: 
```
$ Python odpsolver.py 16 4
Diameter: 3.000
ASPL: 1.750
```
As a result, `regular-graph-o16-d4.gml` is obtained.

Dependency
----------

- numpy
- scipy
- networkx
- joblib
- optuna


Reference
---------

[1] Reiji Hayashi, Tsuyoshi Migita and Norikazu Takahashi, "A genetic algorithm for finding regular graphs with minimum average shortest path length," Proceedings of the 2020 IEEE Symposium Series on Computational Intelligence, pp.-, 2020.

[problem]: http://research.nii.ac.jp/graphgolf/problem.html
