import os
import numpy as np
import networkx as nx
import pygmtools as pygm
import torch
try:
    from torch_geometric.data import Data
except:
    os.system("pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0%2Bcpu.html")
    os.system("pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0%2Bcpu.html")
    os.system("pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0%2Bcpu.html")
    os.system("pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0%2Bcpu.html")
    from torch_geometric.data import Data
from one_hot import one_hot
from torch_geometric.transforms import OneHotDegree
import matplotlib.pyplot as plt
import pygmtools as pygm
pygm.set_backend('pytorch')


######################################################
#                   Constant Variable                #
######################################################

AIDS700NEF_TYPE = [
    'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
    'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
    'Sb', 'Se', 'Ni', 'Te'
]


COLOR = [
    '#FF69B4',  # O - 热情的粉红色
    '#00CED1',  # S - 深蓝绿色
    '#FFD700',  # C - 金色
    '#FFA500',  # N - 橙色
    '#FF6347',  # Cl - 番茄红色
    '#8B008B',  # Br - 深洋红色
    '#00FF7F',  # B - 春天的绿色
    '#40E0D0',  # Si - 绿松石色
    '#FF4500',  # Hg - 橙红色
    '#9932CC',  # I - 深兰花紫色
    '#9370DB',  # Bi - 中紫色
    '#FFA500',  # P - 橙色
    '#FFFF00',  # F - 黄色
    '#B8860B',  # Cu - 深金色
    '#7FFFD4',  # Ho - 碧绿色
    '#FFD700',  # Pd - 金色
    '#B22222',  # Ru - 砖红色
    '#E5E4E2',  # Pt - 浅灰色
    '#A9A9A9',  # Sn - 深灰色
    '#32CD32',  # Li - 酸橙色
    '#CD853F',  # Ga - 秘鲁色
    '#7FFFD4',  # Tb - 碧绿色
    '#8A2BE2',  # As - 紫罗兰色
    '#FFD700',  # Co - 金色
    '#808080',  # Pb - 灰色
    '#A9A9A9',  # Sb - 深灰色
    '#FA8072',  # Se - 鲑鱼色
    '#BEBEBE',  # Ni - 浅灰色
    '#800080'   # Te - 紫色
]


######################################################
#                     Utils Func                     #
######################################################

def from_gexf(filename: str, node_types: list=None):
    r"""
    Read Data from GEXF file
    """
    if not filename.endswith('.gexf'):
        raise ValueError("File type error, 'from_gexf' function only supports GEXF files")
    graph = nx.read_gexf(filename)
    mapping = {name: j for j, name in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    edge_index = torch.from_numpy(np.array(graph.edges, dtype=np.int64).transpose())
    x = None
    labels = None
    data = None
    colors = None
    if 'type' in graph.nodes(data=True)[0].keys():
        labels = dict()
        colors = list()
        num_nodes = graph.number_of_nodes()
        x = torch.zeros(num_nodes, dtype=torch.long)
        node_types = AIDS700NEF_TYPE if node_types is None else node_types
        for node, info in graph.nodes(data=True):
            x[int(node)] = node_types.index(info['type'])
            labels[int(node)] = str(int(node)) + info['type']
            colors.append(COLOR[x[int(node)]])
        x = one_hot(x, num_classes=len(node_types))
        data = Data(x=x, edge_index=edge_index, edge_attr=None)
    return graph, data, labels, colors


def draw(graph, colors, labels, filename, title, pos_type=None):
    if pos_type is None:
        pos = nx.kamada_kawai_layout(graph)
    elif pos_type == "spring":
        pos = nx.spring_layout(graph)
    plt.figure()
    plt.gca().set_title(title)
    nx.draw(graph, pos, with_labels=True, node_color=colors, edge_color='gray', labels=labels)
    plt.savefig(filename)
    plt.clf()


######################################################
#                       GED UI                       #
######################################################

def astar(
    g1_path: str, 
    g2_path: str,
    output_path: str="examples",
    filename: str="example",
    device='cpu'
):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_filename = os.path.join(output_path, filename) + "_{}.png"
    
    # Load data
    g1, d1, l1, c1 = from_gexf(g1_path)
    g2, d2, l2, c2 = from_gexf(g2_path)
    if len(c1) > len(c2):
        graph1, data1, labels1, colors1 = g2, d2, l2, c2
        graph2, data2, labels2, colors2 = g1, d1, l1, c1
    else:
        graph1, data1, labels1, colors1 = g1, d1, l1, c1
        graph2, data2, labels2, colors2 = g2, d2, l2, c2

    # Build Graph and Adj Matrix
    data1 = OneHotDegree(max_degree=6)(data1)
    data2 = OneHotDegree(max_degree=6)(data2)
    feat1 = data1.x.to(device)
    feat2 = data2.x.to(device)
    A1 = torch.tensor(pygm.utils.from_networkx(graph1)).float().to(device)
    A2 = torch.tensor(pygm.utils.from_networkx(graph2)).float().to(device)
    
    # Caculate the ged
    x_pred = pygm.genn_astar(feat1, feat2, A1, A2, return_network=False)
 
    # Plot
    draw(graph1, colors1, labels1, output_filename.format(1), "Graph1")
    draw(graph2, colors2, labels2, output_filename.format(5), f"Graph2")
    
    # Match Process
    total_cost = 0
    labels1_1 = labels1.copy()
    for i in range(x_pred.shape[0]):
        target = torch.nonzero(x_pred[i])[0].item()
        labels1_1[i] = labels1[i].replace(str(i), str(target))
    title = "Node Match"
    draw(graph1, colors1, labels1_1, output_filename.format(2), title)
    
    # Node Change
    cur_cost = 0
    labels1_2 = labels1.copy()
    colors1_2 = colors1.copy()
    target2ori = dict()
    targets = list()
    for i in range(x_pred.shape[0]):
        target = torch.nonzero(x_pred[i])[0].item()
        if labels1_1[i] != labels2[target]:
            cur_cost += 1
        labels1_2[i] = labels2[target]
        colors1_2[i] = colors2[target]
        target2ori[target] = i
        targets.append(target)
    total_cost += cur_cost
    title = f"Node Change"
    draw(graph1, colors1_2, labels1_2, output_filename.format(3), title)
    
    # Edge Change
    leave_cost = np.array(graph2).shape[0] - np.array(graph1).shape[0]
    leave_cost += graph2.number_of_nodes() - graph1.number_of_nodes()
    e2 = np.array(graph2.edges)
    new_edges = list()
    for edge in e2:
        if edge[0] in targets and edge[1] in targets:
            new_edges.append([target2ori[edge[0]], target2ori[edge[1]]])
    graph1.edges = nx.Graph(new_edges).edges
    title = f"Edge Change"
    draw(graph1, colors1_2, labels1_2, output_filename.format(4), title, pos_type="spring")