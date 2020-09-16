import numpy as np
import math
import json
import os
import re

NODE_NUM = 332
NODE_JSON_PATH = '../visualization/data.json'
EDGE_JSON_PATH = '../visualization/link.json'


def read_nodename(filepath):
    ''' 读取节点信息 '''
    nodes_name = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        pattern = re.compile(r'(\d+) "(.*?)"')
        for line in lines:
            match = re.search(pattern, line)
            nodes_name[int(match.group(1))] = match.group(2)
    return nodes_name


def read_edgeinfo(filename, normalize_freq=True):
    ''' 读取边信息 '''
    adjacent = [[math.inf for i in range(NODE_NUM)] for j in range(NODE_NUM)]
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
    return adjacent


def cal_degree(adjacent):
    ''' 计算每个节点的度 '''
    degree = []
    for edges in adjacent:
        degree.append(list(edges).count(1))
    return degree


def generate_node(nodes_name, degree):
    nodes_info = []
    for node_id, node_name in nodes_name.items():
        node_info = {}
        node_info['name'] = node_id
        node_info['category'] = node_classify[node_id]
        node_info['description'] = node_name
        node_info['value'] = degree[node_id-1]
        nodes_info.append(node_info)
    return nodes_info


def generate_edge(adjacent):
    ''' 生成边信息 '''
    edges_info = []
    num = 0
    for i, edges in enumerate(adjacent):
        for j, edge in enumerate(edges):
            if edge == 1:
                num += 1
                edge_info = {}
                edge_info['source'] = i + 1
                edge_info['target'] = j + 1
                edge_info['name'] = 'link' + str(num)
                edge_info['description'] = 'edge'
                edges_info.append(edge_info)
    return edges_info


def save_as_json(information, filename, func):
    ''' 保存成 json 文件 '''
    json_str = json.dumps(information, ensure_ascii=False, indent=4)
    with open(filename, 'w') as f:
        f.write(func + '(' + json_str + ')')


if __name__ == '__main__':
    # 读取数据
    root = '../data'
    node_path = os.path.join(root, 'node_info.txt')
    edge_path = os.path.join(root, 'inf-USAir97.mtx')
    nodes_name = read_nodename(node_path)
    adjacent = read_edgeinfo(edge_path)
    # 类别信息
    partition = np.load('../results/community_partition.npy', allow_pickle=True)
    print(f'partition: {len(partition)}')
    node_classify = [0 for i in range(NODE_NUM+1)]
    for index, parts in enumerate(partition):
        for part in parts:
            node_classify[part] = index
    # 计算信息
    degree = cal_degree(adjacent)
    nodes_info = generate_node(nodes_name, degree)
    edges_info = generate_edge(adjacent)
    # 保存 json 格式的文件
    node_json_path = os.path.join(root, 'node_info.json')
    edge_json_path = os.path.join(root, 'edge_info.json')
    save_as_json(nodes_info, node_json_path, 'get_node')
    save_as_json(edges_info, edge_json_path, 'get_edge')
