import math
import numpy as np
from matplotlib import pyplot as plt
import os


def read_file(filename, normalize_freq=True):
    # 读出node-edge，并存储在邻接矩阵中；该邻接矩阵先不考虑边的权值（即机场距离）；0代表自己与自己，math.inf代表不邻接
    # 注意，文件中节点编号从1开始
    adjacent = [[math.inf for i in range(332)] for j in range(332)]
    freq = [[0 for i in range(332)] for j in range(332)]
    freq_normalized = []
    with open(filename) as f:
        lines = f.readlines()
        lines = lines[24:]
        for ind, line in enumerate(lines):
            edge_info = line.split()
            source = int(edge_info[0])
            target = int(edge_info[1])
            adjacent[source-1][target-1] = 1
            adjacent[target-1][source-1] = 1
            if ind < 332:
                adjacent[ind][ind] = 0
            freq_normalized.append(float(edge_info[2]))
        if normalize_freq == True:
            tmp_max = max(freq_normalized)
            tmp_min = min(freq_normalized)
            for index, f in enumerate(freq_normalized):
                freq_normalized[index] = float((f-tmp_min))/(tmp_max-tmp_min)
        for ind, line in enumerate(lines):
            edge_info = line.split()
            source = int(edge_info[0])
            target = int(edge_info[1])
            freq[source-1][target-1] = freq_normalized[ind]
            freq[target-1][source-1] = freq_normalized[ind]
    return adjacent, freq, freq_normalized


def get_degree(adjacent, topk=10):
    # 算出每个节点的度数并画出柱状图；算出最大度和最小度，并画出度的分布图；画图的函数定义为draw_degree()
    # topk代表选出topk个度数最大的点，获取其编号和度数,储存在topk_biggest_degree_nodes；one_degree_nodes看一看有多少个节点度数为1
    # 返回类型为字典的时候，字典的键为节点编号，从1开始！！！！！返回类型为列表时，下标从0开始
    # 注意使用map函数！！！！！
    degree_dict = {}
    one_degree_nodes = []
    for ind, node in enumerate(adjacent):
        ones = list(map(str, node))
        # 节点编号从1开始
        degree_dict[str(ind+1)] = ''.join(ones).count('1')

    tmp = degree_dict.items()
    sorted_reverse_degree_dict = sorted(
        tmp, key=lambda v: (v[1], v[0]), reverse=True)
    topk_biggest_degree_nodes = sorted_reverse_degree_dict[:topk]
    for it in tmp:
        if it[1] == 1:
            one_degree_nodes.append(it)

    degree_each_node = []
    # degree_each_node 按照节点顺序（1号，2号，3号...储存其度数）
    for t in tmp:
        degree_each_node.append(t[1])

    max_degree = max(degree_each_node)
    min_degree = min(degree_each_node)

    degree_distribution = []
    # 数出度数为1的节点数量，度数为2的节点数量...为画分布图作准备
    for r in range(min_degree, max_degree+1):
        temp = list(map(str, degree_each_node))
        degree_distribution.append(temp.count(str(r)))

    return [degree_dict, topk_biggest_degree_nodes, one_degree_nodes, degree_each_node, degree_distribution]


def floyd_easy(W):
    # 计算全成对最短距离
    n = len(W)
    D = W[:]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i][j] = min(D[i][j], D[i][k]+D[k][j])
    return D


def get_avg_path_length(adjacent, index=False):
    # 计算出avg_(shortest)_path_length以及网络直径
    # adjacent可以是不考虑距离，也可以是考虑距离的
    path_length = 0
    D = floyd_easy(adjacent)
    diameter = np.amax(np.array(D))
    n = len(adjacent)
    for i in range(n):
        for j in range(i+1, n):
            path_length += D[i][j]
    avg_path_length = 2*float(path_length)/(n*(n-1)) if n > 1 else 0
    if index == False:
        return diameter, avg_path_length
    else:
        return np.unravel_index(np.array(D).argmax(), np.array(D).shape)


def draw_degree(inp, topk, draw_distribution=True):
    # 当第三个参数为真的时候，画分布图；否则按顺序画每个节点度的柱状图
    y = np.array(inp)
    if draw_distribution:
        x = np.array([i for i in range(0, len(inp))])
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Node Number")
        plt.bar(x, y)
        for xx, yy in zip(x, y):
            plt.text(xx, yy+1, str(yy), ha='center', va='bottom', fontsize=15)
            break
        plt.savefig('../results/Degree_Distribution.png')
        plt.show()
    else:
        x = np.array([i+1 for i in range(0, len(inp))])
        plt.title("Degree Of Each Node")
        plt.xlabel("Node Index")
        plt.ylabel("Degree")
        plt.bar(x, y)
        for i in range(len(topk)):
            string = 'node '+str(topk[i][0])+': '+str(topk[i][1])
            plt.text(int(topk[i][0])-10, topk[i][1]+4, string, fontsize=10)
        plt.savefig('../results/Degree_Each_Node.png')
        plt.show()


def fitting_func(x, r, b):
    return r*x+b


def draw_degree_with_curve_fitting(inp):
    # 模拟power_law定律
    new_inp = [element for element in inp if element >= 1]
    y = np.array(new_inp)
    y = np.log(y)
    x = np.array([i for i in range(1, len(new_inp)+1)])
    x = np.log(x)
    plt.title("Power Law Degree Distribution")
    plt.xlabel("Logged-Degree")
    plt.ylabel("Logged-Nodes Of This Degree")
    plt.scatter(x, y)
    popt, pcov = curve_fit(fitting_func, x, y, maxfev=1000000)
    yy = [fitting_func(i, popt[0], popt[1]) for i in x]
    plt.plot(x, yy, 'r-', label='fit')
    print('coeff r= ', popt[0], ' coeff b= ', popt[1])
    plt.show()


def clustering_coeff(adjacent):
    cluster_coeff_each = []
    for i in range(332):
        triangle_complete = 0
        triangle = 0
        for j in range(332):
            for k in range(332):
                if adjacent[i][j] == 1 and adjacent[i][k] == 1 and i != j and i != k and j != k:
                    triangle += 1
                    if adjacent[j][k] == 1:
                        triangle_complete += 1
        if triangle == 0:
            cluster_coeff_each.append(0.0)
        else:
            cluster_coeff_each.append(float(triangle_complete)/triangle)
    global_cluster_coeff = np.average(np.array(cluster_coeff_each))

    cluster_dict = {}
    for ind, cluster in enumerate(cluster_coeff_each):
        cluster_dict[str(ind+1)] = cluster

    '''
    sorted_cluster_dict=sorted(cluster_dict.items(),key=lambda x: x[1],reverse=True)
    topk_cluster_dict=sorted_cluster_dict[:topk]
    '''

    return [global_cluster_coeff, cluster_coeff_each, cluster_dict]


def draw_cluster():
    pass


def get_node_name_distance(filename, rate=9000):
    # 带距离的邻接矩阵
    # rate代表距离要乘的系数，考虑到小数距离太小；英里为单位
    D = [[math.inf for i in range(332)] for j in range(332)]
    with open(filename) as f:
        useful_line = []
        lines = f.readlines()
        for index, line in enumerate(lines):
            line = line.split('"')
            for ind, element in enumerate(line):
                line[ind] = element.strip()
            useful_line.append([line[0], line[1]])
            tmp = line[2].split()
            useful_line[index].extend(tmp)
        for i in range(332):
            for j in range(i, 332):
                if i == j:
                    D[i][j] = float(0.0)
                else:
                    D[i][j] = rate*sqrt((float(useful_line[i][2])-float(useful_line[j][2]))**2+(
                        float(useful_line[i][3])-float(useful_line[j][3]))**2)
                    D[j][i] = rate*sqrt((float(useful_line[i][2])-float(useful_line[j][2]))**2+(
                        float(useful_line[i][3])-float(useful_line[j][3]))**2)

    return useful_line, D

# get_node_name_distance('node_info.txt')


if __name__ == '__main__':
    root = '../data'
    filename = os.path.join(root, 'inf-USAir97.mtx')
    adjacent, freq, freq_normalized = read_file(filename)
    diameter, avg_path_length = get_avg_path_length(adjacent)
    cluster_result = clustering_coeff(adjacent)
    degree_result = get_degree(adjacent)
    draw_degree(degree_result[3], degree_result[1])
