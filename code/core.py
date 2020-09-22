from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Applying linkedlist instead of adjacent matrix!


def addEdge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)
    return graph


def get_degree(G):
    degree = {}
    for i in G:
        degree[i] = len(G[i])
    return degree


def core_number(G):
    degrees = get_degree(G)
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}

    core = degrees
    # we can have an initial guess since core always <=degree
    for v in nodes:
        for u in G[v]:
            if core[u] > core[v]:
                G[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core


def find_kcores(G):

    k_cores = {}
    highest_kcore = 0
    core_n = core_number(G)

    for node, k_core in core_n.items():

        if highest_kcore < k_core:
            # keep track of the highest k-core
            highest_kcore = k_core
        if k_core in k_cores:
            k_cores[k_core].append(node)
        else:
            k_cores[k_core] = [node]

    return highest_kcore, k_cores


def draw_kcores(highest_k, k_cores):
    x = np.array(sorted([i for i in k_cores]))
    y = []
    for j in x:
        y.append(len(k_cores[j]))
    y = np.array(y)
    print(f'x: {list(x)}')
    print(f'y: {list(y)}')
    plt.title("Coreness Distribution")
    plt.xlabel("Coreness")
    plt.ylabel("Node Number")
    plt.plot(x, y)
    plt.savefig('../results/Coreness_Distribution.png')
    plt.show()


if __name__ == "__main__":
    graph = defaultdict(list)
    with open('../data/inf-USAir97.mtx', "r") as f:
        edges = f.readlines()
        edges = edges[24:]
        for e in edges:
            nodes = e.split()
            graph = addEdge(graph, nodes[0], nodes[1])
    highest_k, k_cores = find_kcores(graph)
    draw_kcores(highest_k, k_cores)
    # print(highest_k)
    # print((k_cores[26]))
