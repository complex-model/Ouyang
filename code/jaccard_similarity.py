from core import addEdge
from collections import defaultdict
import numpy as np

def jaccard_sim(graph,node1,node2):
    #有相同邻居的节点相似
    upper=len(set(graph[node1]) & set(graph[node2]))
    lower=len(set(graph[node1]) | set(graph[node2]))
    return float(upper/lower)

def save_jaccard(graph,topk=150):
    jaccard_dict={}
    for i in range(1,332):
        for j in range(i+1,333):
            jaccard_dict[(i,j)]=jaccard_sim(graph,i,j)
    np.save('jaccard_similarity_dict.npy',jaccard_dict)
    sorted_jaccard=sorted(jaccard_dict.items(),key=lambda x:x[1],reverse=True)
    #print(sorted_jaccard[:topk])
    np.save('top_150_similar_nodes_pair.npy',sorted_jaccard)

if __name__ == "__main__":
    graph=defaultdict(list)
    with open('inf-USAir97.mtx', "r") as f:
        edges=f.readlines()
        edges=edges[24:]
        for e in edges:  
            nodes=e.split()
            graph=addEdge(graph,int(nodes[0]),int(nodes[1]))
    save_jaccard(graph)