import scipy.sparse as sp
import numpy as np
import networkx as nx

adj = np.load('data/adj.npy')
adj_csr = sp.csr_matrix(adj)

def conductance(adj_csr):
    cond = []
    for i, row in enumerate(adj_csr):
        degrees = 0
        outside = 0
        for j, num in enumerate(row.indices):
            degrees += len(adj_csr[num].indices)
            outside += len(set(adj_csr[num].indices) - set(row.indices))
        if degrees != 0:
            cond.append(outside/degrees)
        else:
            cond.append(1)

    minimas = []
    for i, row in enumerate(adj_csr):
        if len(row.indices) == 0:
            continue
        tmp = min(np.array(cond)[row.indices])
        if cond[i] < tmp:
            minimas.append(i)

    A = np.zeros((adj_csr.shape[0], len(minimas) + 1))
    for i, mini in enumerate(minimas):
        A[adj_csr[mini].indices, i+1] = 1
    A[:, 0] = 1

    return A




