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
    print(f'连通子图的个数为 {cnt}, 各节点所属的子图编号为 {id}')
    number, size = Counter(id).most_common(1)[0]
    largest_subgraph = matrix[:]
    for index, num in enumerate(id):
        assert num != -1
        if num != number:
            # 删除该节点
            largest_subgraph = np.delete(largest_subgraph, index, 0)
            largest_subgraph = np.delete(largest_subgraph, index, 1)
    print(f'size of largest subgraph: {size}')
    return largest_subgraph, size


def random_attack(matrix):
    ''' 随机删除一些点
    :param matrix: 邻接矩阵
    :return: 每次删除节点后，图的鲁棒性
    '''
    mat = matrix[:]
    removed_num, subgraph_size, subgraph_length = [], [], []
    num = 0
    while len(mat) > 1:
        i = random.randint(0, len(mat)-1)
        mat = np.delete(mat, i, axis=0)
        mat = np.delete(mat, i, axis=1)
        num += 1
        removed_num.append(num)  # 记录删除节点的数目
        largest_subgraph, size = cal_largest_subgraph(mat)
        subgraph_size.append(size)  # 记录删除节点后，图的最大连通子图的 size
        _, avg_path_length = model_property.get_avg_path_length(largest_subgraph)
        subgraph_length.append(avg_path_length)  # 记录删除节点后，图的 average path length
    return removed_num, subgraph_size, subgraph_length


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
    # largest_subgraph, size = cal_largest_subgraph(adjacent)
    # 随机删除
    removed_num, subgraph_size, subgraph_length = random_attack(adjacent)
    draw(removed_num, subgraph_size, 'Random', ('node removed', 'Size of the largest subgraph'), '../results/Random_attack_Size.jpg')
    draw(removed_num, subgraph_length, 'Random', ('node removed', 'Average path length'), '../results/Random_attack_Length.jpg')
