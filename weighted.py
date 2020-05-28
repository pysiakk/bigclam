"""
 of the bigClAM algorithm.
Implementation
Throughout the code, we will use tho following variables

  * F refers to the membership preference matrix. It's in [NUM_PERSONS, NUM_COMMUNITIES]
   so index (p,c) indicates the preference of person p for community c.
  * A refers to the adjency matrix, also named friend matrix or edge set. It's in [NUM_PERSONS, NUM_PERSONS]
    so index (i,j) indicates is 1 when person i and person j are friends.
"""

import numpy as np
import pickle
import pandas as pd
from util.generate_data import Datagen, gen_json
import json
import networkx as nx
from scipy import sparse
import scipy


def sigm(x):
    return np.exp((-1) * x) / (1 - np.exp((-1) * x))


def conductance(adj_csr):
    cond = []
    for i, row in enumerate(adj_csr):
        degrees = 0
        outside = 0
        for j, num in enumerate(row.indices):
            degrees += len(adj_csr[num].indices)
            outside += len(set(adj_csr[num].indices) - set(row.indices))
        if degrees != 0:
            cond.append(outside / degrees)
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
        A[adj_csr[mini].indices, i + 1] = 1
    A[:, 0] = 1

    return np.array(A)


def F_init(cond, K):
    if cond.shape[1] >= K:
        return cond[:, 0:K]
    else:
        return sparse.csr_matrix(np.hstack((cond, np.zeros((cond.shape[0], K - cond.shape[1])))))


# def log_likelihood(F, A):
#
#     dotproduct =


def gradient(F, A, i):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf

      * i indicates the row under consideration
    """
    N, C = F.shape

    sum_neigh = np.zeros(C)
    for nb in range(A.shape[1]):
        # tmp = np.zeros(len(F[nb]))
        if A[i, nb] == 0:
            sum_neigh -= F[nb]
        else:
            sum_neigh += A[i, nb] * F[nb] / (np.dot(F[i], F[nb].transpose())[0, 0]) - F[nb]

    grad = sum_neigh
    return grad


def train(A, C, iterations=10):
    N = A.shape[0]
    cond = conductance(A)
    # F = F_init(cond, C)
    F = sparse.csr_matrix(np.random.rand(N, C))
    for n in range(iterations):
        diff = np.ones(N)
        for person in range(N):
            grad = gradient(F, A, person)
            # print(grad)

            F[person] += 0.00001 * grad

            for i, ind in enumerate(F[person].indices):
                if F[person, ind] < 0:
                    F[person, ind] = 0  # F should be nonnegative
        # ll = log_likelihood(F, A)
        # print('At step %5i/%5i ll is %5.3f'%(n, iterations, ll))
        print(n)
    return F


if __name__ == "__main__":
    adj = np.load('data/adj.npy')
    p2c = pickle.load(open('data/p2c.pl', 'rb'))
    amazon = nx.read_edgelist("../communities/email-Eu-core.txt")
    edges = pd.read_csv("../communities/soc-sign-bitcoinotc.csv", delimiter=",", header=None).iloc[:, 0:3]
    # print(edges.head)
    edges.iloc[:,2] = edges.iloc[:,2] + 11
    amazon = nx.from_pandas_edgelist(edges, source=0, target=1, edge_attr=2)
    # adj = nx.adjacency_matrix(amazon)
    adj_csr = sparse.csr_matrix(adj)
    E = len(sparse.find(adj_csr >= 1)[0])//2
    V = adj_csr.shape[0]
    eps = 2 * E / V / (V-1)

    for i in range(10, 11):
        print("Training")
        F = train(adj_csr, i)
        # print(F)
        Z = sparse.find(F > np.sqrt(-np.log(1 - eps)))
        com = [[] for _ in range(max(Z[1]))]
        print("****************************")
        for i in range(max(Z[1])):
            for j in np.where(Z[1] == i)[0]:
                com[i].append(Z[0][j])
        print(com)
        for i, row in enumerate(F):
            print(row.toarray())
        # print(log_likelihood(F, adj_csr))