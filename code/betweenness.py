from collections import defaultdict
from core import addEdge,get_degree
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx

def shortest_path(graph,start,end):
    #BFS
    traversed=[]
    total=[]
    queue=[]
    queue.append([start])
    while len(queue)>0:
        temp_node=queue.pop(0)
        #BFS 节点加入队列，然后按照先进先出的原则
        node=temp_node[-1]
        path=temp_node
        if node not in traversed:
        #未被遍历到，即“白色”的节点
            in_nodes=graph[node]
            for i in range(len(in_nodes)):
                new_path=[]
                new_path.extend(path)
                new_path.append(in_nodes[i])
                queue.append(new_path)
                if in_nodes[i]==end:
                    total.append(new_path)
            traversed.append(node)
            #该节点已经被遍历

    #比较路径长短
    i=0
    if len(total)>1:
        minimum=len(total[0])
        while i<len(total):
            if len(total[i])<minimum:
                minimum=len(total[i])
                i+=1
            elif len(total[i])==minimum:
                i+=1
            elif len(total[i])>minimum:
                total.pop(i)
    return total

def pairs(v):
    #对于节点列表，输出所有可能的边(无重复)，即点-点pair
    all_pairs=[]
    for i in range(len(v)):
        for j in range(i+1,len(v)):
            all_pairs.append([v[i],v[j]])
    return all_pairs

def betweenness_centrality(node_index,possible_pairs,graph,vertices):
    i=0
    while i<len(possible_pairs):
        if node_index in possible_pairs[i]:
            possible_pairs.pop(i)
        else:
            i+=1
    paths={}
    for i in range(len(possible_pairs)):
        paths[possible_pairs[i][0],possible_pairs[i][1]]=(shortest_path(graph,possible_pairs[i][0],possible_pairs[i][1]))
    list1={}
    total=0
    for i in range(len(possible_pairs)):
        count1=len(paths[possible_pairs[i][0],possible_pairs[i][1]])
        count2=0
        for j in range(len(paths[possible_pairs[i][0],possible_pairs[i][1]])):
            tmp=[]
            tmp.extend(paths[possible_pairs[i][0],possible_pairs[i][1]][j])
            if node_index in tmp:
                count2+=1
        list1[possible_pairs[i][0],possible_pairs[i][1]]=float(count2/count1)
        total+=float(count2/count1)
    return total

def create_networkx_graph(graph,vertices):
    #try to use networkx
    edges=[]
    for i in range(len(vertices)-1):
        for j in range(i+1,len(vertices)):
            if j in graph[i]:
                edges.append((i,j))
    G=nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    return G

def node_betweenness(G,topk=15):
    #calculate node_betweenness using networkx
    #G is networkx type graph
    nodes_bet=nx.betweenness_centrality(G,normalized=True)
    np.save('node_betweenness.npy',nodes_bet)
    sorted_nodes_bet=sorted(nodes_bet.items(),key=lambda x:x[1],reverse=True)
    print(sorted_nodes_bet[:topk])
    return nodes_bet

def edge_betweenness(G):
    #calculate edge_betweenness using networkx
    edges_bet=nx.edge_betweenness_centrality(G,normalized=True)
    np.save('edge_betweenness.npy',edges_bet)
    return edges_bet

if __name__ == "__main__":
    graph=defaultdict(list)
    with open('inf-USAir97.mtx', "r") as f:
        edges=f.readlines()
        edges=edges[24:]
        for e in edges:  
            nodes=e.split()
            graph=addEdge(graph,int(nodes[0]),int(nodes[1]))
    vertices=[i for i in range(1,333)]

    G=create_networkx_graph(graph,vertices)
    edges_bet=edge_betweenness(G)

    #y=np.load('node_betweenness.npy',allow_pickle=True)

    '''
    betweenness_dict={}
    i=0
    while i<len(vertices):
        all_pairs=pairs(vertices)
        betweenness_dict[vertices[i]]=betweenness_centrality(vertices[i],all_pairs,graph,vertices)
        i+=1
    sorted_betweenness=sorted(betweenness_dict.items(),key=lambda x:x[1],reverse=True)
    print(sorted_betweenness)
   
    np.save('betweenness_dict_unsorted.npy',betweenness_dict) 
    np.save('betweenness_dict_sorted.npy',sorted_betweenness) 
    '''
    
