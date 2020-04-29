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
import scipy
import pickle
from util.generate_data import Datagen, gen_json
import json
import networkx as nx
import scipy as sp


def sigm(x):
    return np.divide(np.exp(-1.*x), 1.-np.exp(-1.*x))


def log_likelihood(F, A):
    """implements equation 2 of 
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
    A_soft = F.dot(F.T)

    # Next two lines are multiplied with the adjacency matrix, A
    # A is a {0,1} matrix, so we zero out all elements not contributing to the sum
    FIRST_PART = A*np.log(1.-np.exp(-1.*A_soft))
    sum_edges = np.sum(FIRST_PART)
    SECOND_PART = (1-A)*A_soft
    sum_nedges = np.sum(SECOND_PART)

    log_likeli = sum_edges - sum_nedges
    return log_likeli


def gradient(F, A, i):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    
      * i indicates the row under consideration
    
    The many forloops in this function can be optimized, but for
    educational purposes we write them out clearly
    """
    N, C = F.shape

    # print(A == 1)
    # print(A)
    #
    # neighbours = np.where(A == 1)
    # nneighbours = sp.where(A == 0)
    #
    # print(neighbours)
    #
    # sum_neigh = np.zeros((C,))
    # for nb in neighbours[0]:
    #     dotproduct = F[nb].dot(F[i])
    #     sum_neigh += F[nb]*sigm(dotproduct)
    #
    # sum_nneigh = np.zeros((C,))
    # # Speed up this computation using eq.4
    # for nnb in nneighbours[0]:
    #     sum_nneigh += F[nnb]
    #
    # grad = sum_neigh - sum_nneigh
    # return grad

    sum_neigh = np.zeros((C,))
    sum_nneigh = np.zeros((C,))
    for n in range(N):
        dotproduct = F[n].dot(F[i])
        sum_neigh += F[n] * sigm(dotproduct)
        sum_nneigh += F[n]
    grad = sum_neigh - sum_nneigh
    return grad



def train(A, C, iterations=10000):
    # initialize an F
    # print(A.ndim)
    N = A.shape[0]
    F = np.random.rand(N, C)

    for n in range(iterations):
        for person in range(N):
            grad = gradient(F, A, person)

            F[person] += 0.005*grad

            F[person] = np.maximum(0.001, F[person])    # F should be nonnegative
        ll = log_likelihood(F, A)
        print('At step %5i/%5i ll is %5.3f'%(n, iterations, ll))
    return F


if __name__ == "__main__":
    adj = np.load('data/adj.npy')
    p2c = pickle.load(open('data/p2c.pl', 'rb'))
    # generate data
    # datagen = Datagen(40, [.3, .3, .2, .2],[.2, .3, .3, .2] , .1).gen_assignments().gen_adjacency()
    # p2c = datagen.person2comm
    # adj = datagen.adj
    # amazon = nx.read_edgelist("/home/pysiakk/Desktop/communities/com-amazon.ungraph.txt")
    # adj = nx.adjacency_matrix(amazon)
    adj_csr = sp.sparse.csr_matrix(adj)
    print(p2c)

    F = train(adj_csr, 4)
    F_argmax = np.argmax(F, 1)
    data = gen_json(adj, p2c, F_argmax)

    # with open('../data/data.json', 'w') as f:
    with open('ui/assets/data.json', 'w') as f:
        json.dump(data, f, indent=4)

    for i, row in enumerate(F):
        print(row)
        print(p2c[i])

