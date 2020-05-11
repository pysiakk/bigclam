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
from util.generate_data import Datagen, gen_json
import json
import networkx as nx
from scipy import sparse


def sigm(x):
    return np.exp((-1)*x) / (1 - np.exp((-1)*x))


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

    return np.array(A)


def F_init(cond, K):
    if cond.shape[1] >= K:
        return cond[:,0:K]
    else:
        return sparse.csr_matrix(np.hstack((cond, np.zeros((cond.shape[0], K - cond.shape[1])))))


# def log_likelihood(F, A):
#     # """implements equation 2 of
#     # https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
#     # A_soft = F.dot(F.T)
#     #
#     # # Next two lines are multiplied with the adjacency matrix, A
#     # # A is a {0,1} matrix, so we zero out all elements not contributing to the sum
#     # print(A_soft)
#     # FIRST_PART = A*np.log(-np.expm1(-1.*A_soft))
#     # sum_edges = np.sum(FIRST_PART)
#     # sum_nedges = np.sum(A_soft) - A
#     #
#     # log_likeli = sum_edges - sum_nedges
#     # return log_likeli
#
#     for i in range(F.shape[0]):
#         FIRST_PART = []
#         for neigh in  A[i].indices:
#             print(np.expm1(-F[i] @ F[neigh].transpose()))
#             FIRST_PART.append(np.log(- np.expm1(-F[i] @ F[neigh].transpose())))
#     print(FIRST_PART)


def gradient(F, A, i):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    
      * i indicates the row under consideration
    """
    N, C = F.shape

    neighbours = A[i].indices

    sum_neigh = np.zeros(C)
    for nb in neighbours:
        dotproduct = F[i].dot(F[nb].transpose())
        sum_neigh += F[nb]*sigm(dotproduct[0, 0])

    sum_nneigh = np.sum(F, axis=0) - F[i] - np.sum(F[A[i].indices], axis=0)
    grad = sum_neigh - sum_nneigh
    return grad



def train(A, C, iterations=10):
    # initialize an F
    # print(A.ndim)
    N = A.shape[0]
    cond = conductance(A)
    # F = F_init(cond, C)
    F = sparse.csr_matrix(np.random.rand(N,C))
    for n in range(iterations):
        for person in range(N):
            grad = gradient(F, A, person)
            # print(grad)

            F[person] += 0.005*grad

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
    adj = nx.adjacency_matrix(amazon)
    print(adj[1:10].toarray())
    adj_csr = sparse.csr_matrix(adj)
    # print(adj)

    print("Training")
    F = train(adj_csr, 4)
    for i, row in enumerate(F):
        print(row)
