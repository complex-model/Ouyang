from model_property import get_node_name_distance,get_degree
import numpy as np
import unittest
import math
'''
    Implements the Louvain method.
    Input: a weighted undirected graph
    Ouput: a (partition, modularity) pair where modularity is maximum
'''
class PyLouvain:

    @classmethod
    def from_file(cls,distance_matrix_path):
        _,D=get_node_name_distance(distance_matrix_path,1)
        nodes = {}
        edges = []
        for i in range(len(D)):
            nodes[str(i+1)]=1
            for j in range(len(D)):
                if D[i][j]!=0.0 and D[i][j]!=math.inf:
                    edges.append(((str(i+1),str(j+1)),float(1/D[i][j])))
       
        # rebuild graph 
        nodes_,edges_=in_order(nodes,edges)
        return cls(nodes_,edges_)

    '''
        Initializes the method.
        _nodes: a list of ints
        _edges: a list of ((int, int), weight) pairs
    '''
    def __init__(self, nodes, edges):
        self.nodes=nodes
        self.edges=edges
        # precompute m (sum of the weights of all links in network)
        #            k_i (sum of the weights of the links incident to node i)
        self.m=0
        self.k_i=[0 for n in nodes]
        self.edges_of_node={}
        self.w=[0 for n in nodes]
        for e in edges:
            #there's no self-loop initially
            self.m+=e[1]
            self.k_i[e[0][0]]+=e[1]
            self.k_i[e[0][1]]+=e[1] 
            #save edges by node
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]]=[e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]]=[e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        #access community of a node in O(1) time
        self.communities=[n for n in nodes]
        self.actual_partition=[]

    def apply_method(self):
        #apply louvain method
        network=(self.nodes,self.edges)
        best_partition=[[node] for node in network[0]]
        best_q=-1
        i=1
        while 1:
            print("pass #%d" % i)
            i+=1
            partition = self.first_phase(network)
            q=self.compute_modularity(partition)
            partition = [c for c in partition if c]
            print("%s (%.8f)" % (partition,q))
            #clustering initial nodes with partition
            if self.actual_partition:
                actual=[]
                for p in partition:
                    part=[]
                    for n in p:
                        part.extend(self.actual_partition[n])
                    actual.append(part)
                self.actual_partition=actual
            else:
                self.actual_partition=partition
            if q==best_q:
                break
            network=self.second_phase(network,partition)
            best_partition=partition
            best_q=q
        return (self.actual_partition,best_q)

    def compute_modularity(self, partition):
        q=0
        m2=(self.m)*2
        for i in range(len(partition)):
            q+=(0.0+self.s_in[i])/m2 - (0.0+(self.s_tot[i])/m2)**2
        return q

    def compute_modularity_gain(self, node, c, k_i_in):
        #node is an index
        #c is the community index
        #k_i_in is the sum of the weights of the links from node to nodes in c
        return 2*k_i_in-self.s_tot[c]*self.k_i[node]/self.m

    def first_phase(self, network):
        #network: a (nodes, edges) pair
        #make initial partition
        best_partition=self.make_initial_partition(network)
        while 1:
            improvement = 0
            for node in network[0]:
                node_community=self.communities[node]
                #default best community is its own
                best_community=node_community
                best_gain=0
                #remove _node from its community
                best_partition[node_community].remove(node)
                best_shared_links=0
                for e in self.edges_of_node[node]:
                    if e[0][0]==e[0][1]:
                        continue
                    if e[0][0]==node and self.communities[e[0][1]]==node_community or e[0][1]==node and self.communities[e[0][0]]==node_community:
                        best_shared_links+=e[1]
                self.s_in[node_community]-=2*(best_shared_links+self.w[node])
                self.s_tot[node_community]-=self.k_i[node]
                self.communities[node]=-1
                communities = {} 
                #only consider neighbors of different communities
                for neighbor in self.get_neighbors(node):
                    community=self.communities[neighbor]
                    if community in communities:
                        continue
                    communities[community]=1
                    shared_links=0
                    for e in self.edges_of_node[node]:
                        if e[0][0]==e[0][1]:
                            continue
                        if e[0][0]==node and self.communities[e[0][1]]==community or e[0][1]==node and self.communities[e[0][0]]==community:
                            shared_links += e[1]
                    #compute modularity gain obtained by moving _node to the community of _neighbor
                    gain = self.compute_modularity_gain(node,community,shared_links)
                    if gain > best_gain:
                        best_community=community
                        best_gain=gain
                        best_shared_links=shared_links
                # insert _node into the community maximizing the modularity gain
                best_partition[best_community].append(node)
                self.communities[node]=best_community
                self.s_in[best_community]+=2*(best_shared_links+self.w[node])
                self.s_tot[best_community]+=self.k_i[node]
                if node_community!=best_community:
                    improvement=1
            if not improvement:
                break
        return best_partition

    def get_neighbors(self, node):
    #yields the nodes adjacent to _node
        for e in self.edges_of_node[node]:
            if e[0][0]==e[0][1]: 
            #a node is not neighbor with itself
                continue
            if e[0][0]==node:
                yield e[0][1]
            if e[0][1]==node:
                yield e[0][0]

    def make_initial_partition(self, network):
        partition=[[node] for node in network[0]]
        self.s_in=[0 for node in network[0]]
        self.s_tot=[self.k_i[node] for node in network[0]]
        for e in network[1]:
            if e[0][0]==e[0][1]: # only self-loops
                self.s_in[e[0][0]]+=e[1]
                self.s_in[e[0][1]]+=e[1]
        return partition

    def second_phase(self, network, partition):
        nodes_=[i for i in range(len(partition))]
        #relabelling communities
        communities_=[]
        d={}
        i=0
        for community in self.communities:
            if community in d:
                communities_.append(d[community])
            else:
                d[community]=i
                communities_.append(i)
                i+=1
        self.communities=communities_
        # building relabelled edges
        edges_={}
        for e in network[1]:
            ci=self.communities[e[0][0]]
            cj=self.communities[e[0][1]]
            try:
                edges_[(ci, cj)]+=e[1]
            except KeyError:
                edges_[(ci, cj)]=e[1]
        edges_=[(k, v) for k,v in edges_.items()]
        # recomputing k_i vector and storing edges by node
        self.k_i=[0 for n in nodes_]
        self.edges_of_node={}
        self.w=[0 for n in nodes_]
        for e in edges_:
            self.k_i[e[0][0]]+=e[1]
            self.k_i[e[0][1]]+=e[1]
            if e[0][0]==e[0][1]:
                self.w[e[0][0]]+=e[1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]]=[e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]]=[e]
            elif e[0][0]!=e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        #resetting communities
        self.communities=[n for n in nodes_]
        return (nodes_, edges_)

'''
    Rebuilds a graph with successive nodes' ids.
    _nodes: a dict of int
    _edges: a list of ((int, int), weight) pairs
'''
def in_order(nodes, edges):
        # rebuild graph with successive identifiers
        nodes=list(nodes.keys())
        nodes.sort()
        i=0
        nodes_=[]
        d={}
        for n in nodes:
            nodes_.append(i)
            d[n]=i
            i+=1
        edges_=[]
        for e in edges:
            edges_.append(((d[e[0][0]],d[e[0][1]]),e[1]))
        return (nodes_,edges_)

class PylouvainTest(unittest.TestCase):

    def test_air_route(self):
        py_lou=PyLouvain.from_file("node_info.txt")
        partition,q=py_lou.apply_method()
        real_partition=[[] for kk in range(len(partition))]
        for index1,p in enumerate(partition):
            for element in p:
                real_partition[index1].append(element+1)     
        #print(real_partition)
        np.save('community_partition.npy',real_partition)

def evaluate_cluster(partition):
    A=np.load('adjacent_matrix_without_distance.npy',allow_pickle=True)
    degree_res=get_degree(A,topk=20)
    topk=degree_res[1]
    topk_nodes=[]
    for i in topk:
        topk_nodes.append(i[0])

    community_with_big_node_count={}
    for kk in range(len(partition)):
        name='community '+str(kk+1)
        community_with_big_node_count[name]=0

    for index,p in enumerate(partition):
        for element in p:
            if str(element) in topk_nodes:
                name_='community '+str(index+1)
                community_with_big_node_count[name_]+=1
    print(community_with_big_node_count)
    return community_with_big_node_count

if __name__ == '__main__':
    unittest.main()
    #p=np.load('community_partition.npy',allow_pickle=True)
    #big_community=evaluate_cluster(p)
