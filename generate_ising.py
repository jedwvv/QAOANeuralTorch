import networkx as nx
import numpy as np

def generate_ising_array(N,prob_edge=1.0, weighted=False, seed=8094):
    rng = np.random.default_rng(seed)
    G = generate_graph(N, prob_edge=prob_edge, seed=seed)
    J = nx.to_numpy_array(G)
    if weighted:
        weights = rng.normal(loc=0, scale=1.0, size=(N,N))
        J = weights * J
    return J, G

def generate_graph(N, prob_edge=1.0, seed=8094):
    G = nx.erdos_renyi_graph(n=N, p=prob_edge, seed=seed)
    return G