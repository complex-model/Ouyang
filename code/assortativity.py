from collections import defaultdict
import numpy as np
from core import addEdge,get_degree

def calculate_assortativity(deg,graph,M=2126):
    #注意到边重复计算了，所以公式中的系数都要再乘0.5
    sigma_j_mul_k=0
    sigma_j_plus_k=0
    sigma_j_2_plus_k_2=0
    for start_node in graph:
        for end_node in graph[start_node]:
            sigma_j_plus_k+=(deg[start_node]+deg[end_node])
            sigma_j_mul_k+=(deg[start_node]*deg[end_node])
            sigma_j_2_plus_k_2+=(deg[start_node]**2+deg[end_node]**2)
    a=float((0.5*sigma_j_mul_k/M)-(0.25*sigma_j_plus_k/M)**2)/((0.25*sigma_j_2_plus_k_2/M)-(0.25*sigma_j_plus_k/M)**2)
    if a>0:
        print('assortativity is ',a,' indicating big node connects big node')
    else:
        print('assortativity is ',a,' indicating big node connects small node')
    return a

if __name__ == "__main__":
    graph=defaultdict(list)
    with open('inf-USAir97.mtx', "r") as f:
        edges=f.readlines()
        edges=edges[24:]
        for e in edges:  
            nodes=e.split()
            graph=addEdge(graph,nodes[0],nodes[1])
    deg=get_degree(graph)
    #print(deg)
    #print(graph)
    a=calculate_assortativity(deg,graph)