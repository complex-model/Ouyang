from collections import Counter
import matplotlib.pyplot as plt
import model_property
import numpy as np
import random
import os


def dfs(matrix, visited, i, cnt, id):
    ''' 深度优先搜索
    :param matrix: 邻接矩阵
    :param visited: 记录节点访问状态
    :param i: 当前访问的节点
    :param cnt: 连通分支的编号
    :param id: 记录每个节点所属的连通分支
    '''
    for j in range(len(matrix)):
        if matrix[i][j] == 1 and visited[j] == 0:
            visited[j] = 1
            id[j] = cnt
            dfs(matrix, visited, j, cnt, id)


def cal_largest_subgraph(matrix):
    ''' 计算最大的连通子图及其 size
    :param matrix: 图的邻接矩阵
    :return: largest subgraph, size of it
    '''
    n = len(matrix)
    visited = [0] * n
    id = [-1] * n
    cnt = 0
    for i in range(n):
        if visited[i] == 0:
            visited[i] = 1
            id[i] = cnt
            dfs(matrix, visited, i, cnt, id)
            cnt += 1
    print(f'连通子图的个数为 {cnt}')
    number, size = Counter(id).most_common(1)[0]
    largest_subgraph = matrix[:]
    remove_index = [index for index, num in enumerate(id) if num != number]
    # 删除该节点
    largest_subgraph = np.delete(largest_subgraph, remove_index, 0)
    largest_subgraph = np.delete(largest_subgraph, remove_index, 1)
    print(f'size of largest subgraph: {size}')
    return largest_subgraph, size


def attack(matrix, random_attack=True, node_skip=1):
    ''' 随机删除一些点
    :param matrix: 邻接矩阵
    :return: 每次删除节点后，图的鲁棒性
    '''
    mat = matrix[:]
    removed_num, subgraph_size, subgraph_length = [], [], []
    num = 0
    while len(mat) > node_skip:
        remove_index = []
        if random_attack:
            remove_index = np.random.choice(range(len(mat)), node_skip)
        else:
            remove_index = get_max_degree_node(mat, top=node_skip)
        mat = np.delete(mat, remove_index, axis=0)
        mat = np.delete(mat, remove_index, axis=1)
        num += node_skip
        removed_num.append(num)  # 记录删除节点的数目
        largest_subgraph, size = cal_largest_subgraph(mat)
        subgraph_size.append(size)  # 记录删除节点后，图的最大连通子图的 size
        _, avg_path_length = model_property.get_avg_path_length(largest_subgraph)
        subgraph_length.append(avg_path_length)  # 记录删除节点后，图的 average path length
    return removed_num, subgraph_size, subgraph_length


def get_max_degree_node(matrix, top=1):
    '''获取图中度最大的前 top 个点'''
    degree = []
    for mat in matrix:
        degree.append(list(mat).count(1))
    node_degree = {index: deg for index, deg in enumerate(degree)}
    node_degree = sorted(node_degree.items(), key=lambda v: v[1], reverse=True)
    return [index for index, value in node_degree[:top]]


def draw(x, y, title, label, filename, color='blue'):
    ''' 可视化 '''
    assert len(label) == 2
    x_label, y_label = label
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y, np.pi * 1, c=color, alpha=0.8)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    root = '../data'
    filename = os.path.join(root, 'inf-USAir97.mtx')
    adjacent, freq, freq_normalized = model_property.read_file(filename)
    # 随机删除
    # removed_num, subgraph_size, subgraph_length = attack(adjacent, random_attack=True, node_skip=10)
    # draw(removed_num, subgraph_size, 'Random', ('node removed', 'Size of the largest subgraph'), '../results/Random_attack_Size.jpg')
    # draw(removed_num, subgraph_length, 'Random', ('node removed', 'Average path length'), '../results/Random_attack_Length.jpg')
    # 选取度大的点删除
    removed_num, subgraph_size, subgraph_length = attack(adjacent, random_attack=False, node_skip=10)
    draw(removed_num, subgraph_size, 'Intentional', ('node removed', 'Size of the largest subgraph'), '../results/Intentional_attack_Size.jpg')
    draw(removed_num, subgraph_length, 'Intentional', ('node removed', 'Average path length'), '../results/Intentional_attack_Length.jpg')
