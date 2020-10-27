#! /usr/bin/env python
# coding: utf-8
# file name: odpsolver.py
# Author: Takehiro Sano
# Reference: 


import sys
import random
from operator import attrgetter, itemgetter
import numpy as np
import networkx as nx
from scipy import sparse
import joblib
import optuna


class Individual:
    """個体クラス"""
    def __init__(self, G, fitness=None):
        self.G = G
        self.fitness = fitness


    def copy(self):
        """Indivisualのdeepcopy関数"""
        return Individual(self.G.copy(as_view=False), self.fitness)


def calc_fitness(G):
        """個体の適用度の計算
    
        Parameter
        ----------
        G: グラフ

        Return
        ----------
        score: 個体と対応するグラフの平均頂点間距離
        """
        try: 
            score = nx.average_shortest_path_length(G)
        except NetworkXError:
            score = float('inf')
        return score


def graph2ind(G):
    """グラフから個体への変換
    
    Parameters
    ----------
    G: グラフ

    Return
    ----------
    ind: グラフと対応する個体
    """
    A = nx.adjacency_matrix(G)
    n = A.shape[0]
    ind = [A[i,j] for i in range(n) for j in range(n) if i < j]
    return ind


def init_pop(n_node, n_degree, n_ind):
    """初期個体の生成
    
    Parameters
    ----------
    n_node: 頂点数
    n_degree: 次数
    n_ind: 個体数

    Return
    ----------
    pop: 個体集合
    """
    pop = []

    for i in range(n_ind):
        G = nx.random_regular_graph(d=n_degree, n=n_node)
        ind = Individual(G)
        pop.append(ind)
    
    return pop


def exec_selection(pop, n_ind, tournsize):
    """選択
    
    Parameters
    ----------
    pop: 個体集合
    n_ind: 個体数
    toursize: トーナメントサイズ 

    Return
    ----------
    selection: 選択によって得られる個体集合
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
    """個体の一組に対する1点交叉
    
    Parameters
    ----------
    ind1: 個体1
    ind2: 個体2
    cxpoint: 交差点
    n_degree: 次数
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
    """1点交叉
    
    Parameters
    ----------
    pop: 個体集合
    n_node: 頂点数
    n_degree: 次数
    n_ind: 個体数
    prob_c: 交叉確率
    """
    # 交叉点のリスト
    cxpoints = [np.random.choice(np.arange(n_node), 2, replace=False) for _ in range(n_ind//2)]
    # 一様分布のリスト
    probs = [np.random.rand() for _ in range(n_ind//2)]

    joblib.Parallel(n_jobs=-2, verbose=0)([joblib.delayed(exec_crossover_once)(ind1, ind2, cxpoint, n_degree) for cxpoint, prob, ind1, ind2 in zip(cxpoints, probs, pop[::2], pop[1::2]) if prob < prob_c])


def is_valid_degree_sequence_havel_hakimi(deg_sequence):
    """Havel-Hakamiの定理
    
    Parameters
    ----------
    deg_sequence: 次数列

    Return
    ----------
    グラフ的であるか否かの真偽値 (Boolean)
    """

    deg = np.array(deg_sequence, dtype=int)

    for _ in range(len(deg)):
        deg_sorted = np.sort(deg)[::-1]
        maxv = deg_sorted[0]
        minv = deg_sorted[-1]
        if minv < 0:
            return False
        elif maxv == 0:
            return True
        deg = deg_sorted[1:]
        deg[:maxv] -= 1


def keep_regularity(G, n_degree, random_state=None):
    """正則性の確保
    
    Parameters
    ----------
    G: グラフ
    n_degree: 次数

    Return
    ----------
    G: 次数が n_degree の正則グラフ
    """
    # シードの設定
    np.random.seed(random_state)

    # グラフGから隣接行列Aを生成
    A = nx.adjacency_matrix(G).tolil()
    
    # 次数下げ: 次数がn_degree以下となるまで続ける
    while True:
        deg = np.array(A.sum(axis=1)).flatten()
        if np.max(deg) <= n_degree:
            break
        # Gの最大次数のノードと対応する行を求める
        idx_row = np.argmax(deg)
        # その行の最大要素の添字を求める
        for idx in np.argsort(-deg):
            if idx_row == idx:
                continue
            else:
                if A[idx, idx_row] > 0:
                    idx_col = idx
                    break
        
        # 辺(idx_row, idx_col)を削除
        A[idx_row, idx_col] -= 1
        A[idx_col, idx_row] -= 1

    # Havel-Hakimiの定理による正則性の確保
    while True:
        deg = np.array(A.sum(axis=1)).flatten()
        if np.all(deg == n_degree):
            # 正則性が確保されたときbreak
            break

        # Havel-Hakimiの定理
        if is_valid_degree_sequence_havel_hakimi(n_degree - deg): # グラフ的である場合
            # 辺の追加
            min_deg_idx = np.argmin(deg)
            min_deg_val = int(deg[min_deg_idx])
            # 追加する辺の本数はn_add_edges
            n_add_edges = n_degree - min_deg_val
            
            # Aのmin_deg_idx行目の中で,値の小さいn_add_edges個の要素に1を加え，次数をn_degreeにする
            tmp = list(np.argsort(deg))
            tmp.remove(min_deg_idx) # 自己ループとなる辺の追加を回避
            for i in tmp[:n_add_edges]:
                # A[min_deg_idx, tmp[:n_add_edges]] += 1
                # A[tmp[:n_add_edges], min_deg_idx] += 1
                A[min_deg_idx, i] += 1
                A[i, min_deg_idx] += 1
        else: # グラフ的でない場合
            # 次数がn_degreeとなる頂点と対応する添字を求める
            regular_deg_idx = [i for i, v in enumerate(deg) if v == n_degree]
            while True:
                # 次数がn_degreeの相異なる頂点番号node1_idx, node2_idxを求める
                node1_idx, node2_idx = np.random.choice(regular_deg_idx, 2, replace=False)
                if A[node1_idx, node2_idx] > 0:
                    # 辺が存在するならば削除しbreak
                    A[node1_idx, node2_idx] -= 1
                    A[node2_idx, node1_idx] -= 1
                    break

    # 多重辺の削除
    while True:
        # ある多重辺の行と列の添字を求める
        row, col, _ = sparse.find(A > 1)
        if len(row) == 0:
            # 多重辺が存在しないときbreak
            break

        row = row[0]
        col = col[0]

        x, y, _ = sparse.find(A > 0)
        
        for i,j in zip(x, y):
            if not (i in set([row, col]) or j in set([row, col])) and A[i, row] == 0 and A[j, col] == 0:
                # i, j は辺が存在する二つの頂点を意味する．
                # このとき，i, j が row, col のどちらとも異なっており，かつ
                # 頂点iとrowの間に辺が存在せず，かつ
                # 頂点jとcolの間に辺が存在しないならば，
                # 2-switchによって，(row, col)と(i,j)を(i, row)と(j, col)へ
                #
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

    # 隣接行列AからグラフGを生成
    G = nx.Graph(A)
    
    return G
    

def exec_mutation(pop, prob_m):
    """突然変異
    
    Parameters
    ----------
    pop: 個体集合
    prob_m: 突然変異確率
    """
    for ind in pop:
        if np.random.rand() < prob_m:
            nx.double_edge_swap(ind.G, nswap=1)
            ind.fitness = None


def find_regular_graph_aspl(n_node, n_degree, n_ind, prob_c, prob_m, n_gen, tournsize, random_state=None, verbose=False):
    """3.2節の提案アルゴリズムの実装
    
    Parameters
    ----------
    n_node: 頂点数
    n_degree: 次数
    n_ind: 個体数
    prob_c: 交叉確率
    prob_m: 突然変異確率
    n_gen: 世代数
    tournsize: トーナメントサイズ

    Return
    ----------
    best_ind: 適応度が最小の個体
    """
    # シードの設定
    random.seed(random_state)
    np.random.seed(random_state)

    # 初期個体の生成
    pop = init_pop(n_node, n_degree, n_ind)

    # 評価
    scores = joblib.Parallel(n_jobs=-2, verbose=0)([joblib.delayed(calc_fitness)(ind.G) for ind in pop])
    for ind, score in zip(pop, scores):
        ind.fitness = score

    if verbose:
        print("Generation loop start.")
        best_ind = min(pop, key=attrgetter('fitness'))
        print("[Generation: 0] Best fitness: {}".format(best_ind.fitness))

    for g in range(n_gen):
        # 選択
        offspring = exec_selection(pop, n_ind, tournsize)

        # 交叉（注．offspring は inplace な更新）
        exec_crossover(offspring, n_node, n_degree, n_ind, prob_c)

        # 突然変異（注．offspring は inplace な更新）
        exec_mutation(offspring, prob_m)

        # 世代の更新
        pop = offspring
        tmp = joblib.Parallel(n_jobs=-2, verbose=0)([joblib.delayed( lambda x, y: (x, calc_fitness(y)) )(idx, ind.G) for idx, ind in enumerate(pop) if ind.fitness is None])
        for i, score in tmp:
            pop[i].fitness = score
        
        if verbose:
            best_ind = min(pop, key=attrgetter('fitness'))
            print("[Generation: {}] Best fitness: {}".format(g+1, best_ind.fitness))
    
    if verbose:
        print("Generation loop ended.")

    return best_ind


def optuna_objective(n_node, n_degree, n_ind, n_gen, random_state=None, verbose=False):
    def _optuna_objective(trial):
        prob_c = trial.suggest_discrete_uniform("prob_c", 0.4, 0.8, 0.05)
        prob_m = trial.suggest_discrete_uniform("prob_m", 0.1, 0.3, 0.05)
        tournsize = trial.suggest_int('r', 2, 5) 
        return find_regular_graph_aspl(n_node=n_node, n_degree=n_degree, n_ind=n_ind, prob_c=prob_c, prob_m=prob_m, n_gen=n_gen, tournsize=tournsize, random_state=random_state, verbose=verbose).fitness

    return _optuna_objective


def main():
    n = int(sys.argv[1])      # 頂点数
    k = int(sys.argv[2])       # 次数
    m = 100     # 個体数
    tmax = 250  # 世代数
    random_state = 0
    verbose = True
    n_trials = 100

    # optuna messege on/off
    #optuna.logging.disable_default_handler()
    #optuna.logging.enable_default_handler()

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=123)) 
    study.optimize(optuna_objective(n, k, m, tmax, random_state, verbose), n_trials=100)
    print(study.best_params)

    #ret_ind = find_regular_graph_aspl(n_node=n, n_degree=k, n_ind=m, prob_c=pc, prob_m=pm, n_gen=tmax, tournsize=r, random_state=random_state, verbose=verbose)

    # print('Obtained Individual: ')
    # print(graph2ind(ret_ind.G))
    # print('Fitness: {}'.format(ret_ind.fitness))


if __name__ == "__main__":
    main()