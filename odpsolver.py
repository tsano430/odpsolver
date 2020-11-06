# coding: utf-8

# file name: odpsolver.py
# Author: Takehiro Sano
# License: MIT License


import sys
import random
from operator import attrgetter, itemgetter
import numpy as np
import networkx as nx
from scipy import sparse
import joblib
import optuna


class Individual:
    def __init__(self, G, fitness=None):
        self.G = G
        self.fitness = fitness


    def copy(self):
        """deepcopy"""
        return Individual(self.G.copy(as_view=False), self.fitness)


def calc_fitness(G):
        """calculate fitness
    
        Parameter
        ----------
        G: graph

        Return
        ----------
        score: average shortest path length
        """
        try: 
            score = nx.average_shortest_path_length(G)
        except NetworkXError:
            score = float('inf')
        return score


def init_pop(n_node, n_degree, n_ind):
    """generate initial individuals
    
    Parameters
    ----------
    n_node: number of nodes
    n_degree: number of degree
    n_ind: number of individuals

    Return
    ----------
    pop: set of individuals
    """
    pop = []

    for i in range(n_ind):
        G = nx.random_regular_graph(d=n_degree, n=n_node)
        ind = Individual(G)
        pop.append(ind)
    
    return pop


def exec_selection(pop, n_ind, tournsize):
    """selection
    
    Parameters
    ----------
    pop: set of individuals
    n_ind: number of individuals
    toursize: tournament size

    Return
    ----------
    selection: set of individuals obtained from selection operation
    """
    selection = []

    for _ in range(n_ind):
        shuffled_idx = np.random.choice(np.arange(n_ind), size=tournsize)
        aspirants = itemgetter(*shuffled_idx)(pop)
        best_aspirants = min(aspirants, key=attrgetter('fitness'))
        selection.append(best_aspirants)

    selection = joblib.Parallel(n_jobs=-2, verbose=0)([joblib.delayed(lambda x: x.copy())(ind) for ind in selection])

    return selection


def exec_crossover_once(ind1, ind2, cxpoint, n_degree):
    """single point crossover for two individuals
    
    Parameters
    ----------
    ind1: individual 1
    ind2: individual 2
    cxpoint: crossover point
    n_degree: number of degree
    """
    g1 = ind1.G
    g2 = ind2.G

    x = np.min(cxpoint)
    y = np.max(cxpoint)

    edge1 = set(sorted(ind1.G.edges))
    edge2 = set(sorted(ind2.G.edges))

    edge1_lower = {(i, j) for i, j in edge1 if i < x or (i == x and j <= y)}
    edge1_upper = edge1 - edge1_lower
    edge2_lower = {(i, j) for i, j in edge2 if i < x or (i == x and j <= y)}
    edge2_upper = edge2 - edge2_lower

    g1.remove_edges_from(edge1_lower)
    g1.add_edges_from(edge2_lower)

    g2.remove_edges_from(edge2_lower)
    g2.add_edges_from(edge1_lower)

    ind1.G = keep_regularity(g1, n_degree, random_state=0)
    ind2.G = keep_regularity(g2, n_degree, random_state=0)

    ind1.fitness = None
    ind2.fitness = None


def exec_crossover(pop, n_node, n_degree, n_ind, prob_c):
    """single point crossover
    
    Parameters
    ----------
    pop: set of individuals
    n_node: number of nodes
    n_degree: number of degree
    n_ind: number of individuals
    prob_c: crossover probability
    """
    cxpoints = [np.random.choice(np.arange(n_node), 2, replace=False) for _ in range(n_ind//2)]
    probs = [np.random.rand() for _ in range(n_ind//2)]
    joblib.Parallel(n_jobs=-2, verbose=0)(
        [joblib.delayed(exec_crossover_once)(ind1, ind2, cxpoint, n_degree) 
         for cxpoint, prob, ind1, ind2 in zip(cxpoints, probs, pop[::2], pop[1::2]) if prob < prob_c])


def keep_regularity(G, n_degree, random_state=None):
    """keep regularity of graph
    
    Parameters
    ----------
    G: graph
    n_degree: number of degree

    Return
    ----------
    G: "n_degree"-regular graph
    """
    np.random.seed(random_state)

    # transform graph into adjacency matrix
    A = nx.adjacency_matrix(G).tolil()
    
    while True:
        deg = np.array(A.sum(axis=1)).flatten()
        if np.max(deg) <= n_degree:
            break

        idx_row = np.argmax(deg)
        for idx in np.argsort(-deg):
            if idx_row == idx:
                continue
            else:
                if A[idx, idx_row] > 0:
                    idx_col = idx
                    break
        
        A[idx_row, idx_col] -= 1
        A[idx_col, idx_row] -= 1

    while True:
        deg = np.array(A.sum(axis=1)).flatten()
        if np.all(deg == n_degree):
            break

        # Havel-Hakimi's Theorem
        if nx.is_valid_degree_sequence_havel_hakimi(n_degree - deg): 
            # graphic
            min_deg_idx = np.argmin(deg)
            min_deg_val = int(deg[min_deg_idx])
            n_add_edges = n_degree - min_deg_val
            tmp = list(np.argsort(deg))
            tmp.remove(min_deg_idx) 
            for i in tmp[:n_add_edges]:
                A[min_deg_idx, i] += 1
                A[i, min_deg_idx] += 1
        else: # not graphic
            regular_deg_idx = [i for i, v in enumerate(deg) if v == n_degree]
            while True:
                node1_idx, node2_idx = np.random.choice(regular_deg_idx, 2, replace=False)
                if A[node1_idx, node2_idx] > 0:
                    A[node1_idx, node2_idx] -= 1
                    A[node2_idx, node1_idx] -= 1
                    break

    while True:
        row, col, _ = sparse.find(A > 1)
        if len(row) == 0:
            break

        row = row[0]
        col = col[0]

        x, y, _ = sparse.find(A > 0)
        
        for i,j in zip(x, y):
            if not (i in set([row, col]) or j in set([row, col])) and A[i, row] == 0 and A[j, col] == 0:
                A[row, col] -= 1
                A[col, row] -= 1
                A[i, j] -= 1
                A[j, i] -= 1
                #
                A[i, row] += 1
                A[row, i] += 1
                A[col, j] += 1
                A[j, col] += 1
                break

    # transform adjacency matrix into graph
    G = nx.Graph(A)
    
    return G
    

def exec_mutation(pop, prob_m):
    """mutation
    
    Parameters
    ----------
    pop: set of individuals
    prob_m: mutation probability
    """
    for ind in pop:
        if np.random.rand() < prob_m:
            nx.double_edge_swap(ind.G, nswap=1)
            ind.fitness = None


def odpsolver(n_node, n_degree, n_ind, prob_c, prob_m, n_gen, tournsize, random_state=None, verbose=False):
    """A genetic algorithm for finding regular graphs with minimum ASPL
    
    Parameters
    ----------
    n_node: number of nodes
    n_degree: number of degree
    n_ind: number of individuals
    prob_c: crossover probability
    prob_m: mutation probability
    n_gen: number of generations
    tournsize: tournament size

    Return
    ----------
    best_ind: individual with minimum fitness value
    """
    random.seed(random_state)
    np.random.seed(random_state)

    pop = init_pop(n_node, n_degree, n_ind)

    scores = joblib.Parallel(n_jobs=-2, verbose=0)([joblib.delayed(calc_fitness)(ind.G) for ind in pop])
    for ind, score in zip(pop, scores):
        ind.fitness = score
    best_ind = min(pop, key=attrgetter('fitness'))

    if verbose:
        print("Generation loop start.")
        print("[Generation: 0] Best fitness: {}".format(best_ind.fitness))

    for g in range(n_gen):
        offspring = exec_selection(pop, n_ind, tournsize)
        exec_crossover(offspring, n_node, n_degree, n_ind, prob_c)
        exec_mutation(offspring, prob_m)

        # update next population
        pop = offspring
        tmp = joblib.Parallel(n_jobs=-2, verbose=0)(
            [joblib.delayed( lambda x, y: (x, calc_fitness(y)) )(idx, ind.G) for idx, ind in enumerate(pop) if ind.fitness is None])
        for i, score in tmp:
            pop[i].fitness = score
        best_ind = min(pop, key=attrgetter('fitness'))

        if verbose:
            print("[Generation: {}] Best fitness: {}".format(g+1, best_ind.fitness))
    
    if verbose:
        print("Generation loop ended.")

    return best_ind


def optuna_objective(n_node, n_degree, n_ind, n_gen, random_state=None, verbose=False):
    def _optuna_objective(trial):
        prob_c = trial.suggest_discrete_uniform("prob_c", 0.4, 0.8, 0.05)
        prob_m = trial.suggest_discrete_uniform("prob_m", 0.1, 0.3, 0.05)
        tournsize = trial.suggest_int('r', 2, 5) 
        return odpsolver(n_node=n_node, n_degree=n_degree, n_ind=n_ind, prob_c=prob_c, prob_m=prob_m, 
                         n_gen=n_gen, tournsize=tournsize, random_state=random_state, verbose=verbose).fitness

    return _optuna_objective


def main():
    n = int(sys.argv[1])        # number of nodes
    k = int(sys.argv[2])        # number of degree
    m = 100                     # number of individuals
    tmax = 150                  # number of generations

    random_state = 0
    verbose = False
    n_trials = 10

    # optuna message on/off
    if verbose:
        optuna.logging.enable_default_handler()
    else:
        optuna.logging.disable_default_handler()

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=random_state)) 
    study.optimize(optuna_objective(n, k, m, tmax, random_state, verbose), n_trials=n_trials)
    pc = study.best_params['prob_c']
    pm = study.best_params['prob_m']
    r = study.best_params['r'] # tournament size

    ret_ind = odpsolver(n_node=n, n_degree=k, n_ind=m, prob_c=pc, prob_m=pm, n_gen=tmax, 
                        tournsize=r, random_state=random_state, verbose=verbose)

    # output
    print('Diameter: {:.3f}'.format(nx.diameter(ret_ind.G)))
    print('ASPL: {:.3f}'.format(ret_ind.fitness))
    f = open('regular-graph-o{}-d{}.gml'.format(n, k), 'wb')
    nx.write_gml(ret_ind.G, f)
    f.close()

    
if __name__ == "__main__":
    main()
