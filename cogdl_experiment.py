import dgl
from cogdl import experiment
import numpy as np
from utils import load_data
import networkx as nx
from cogdl.data import Graph
import torch
from random import randint
import joblib
from cogdl.datasets import NodeDataset

np.seterr(divide="ignore", invalid="ignore")

datasets = ["cornell", "texas", "wisconsin"] + ["chameleon"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ratio = 0.6
baseline_model = "gcn"  # "mlp","gcn","gat","mixhop", “gcnii”, "ppnp","dropedge_gcn","grand"，"autognn"


def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        if output[i]:
            nodes = output[i]
        else:
            output.pop(i)
            break
    return output


def get_one_homophilious_node(target_node, K_neigbors, model):
    for k_neighors_list in K_neigbors.values():
        for to_node in k_neighors_list:
            if to_node == target_node:
                continue
            if model.predict(
                torch.cat((features[target_node], features[to_node])).unsqueeze(0)
            ).item():
                return to_node
    return target_node


def processed_g_by_rewiring(g, model):
    preprocessed_g = nx.Graph()
    original_g = g.to_networkx()

    for (u, v) in original_g.edges():
        if model.predict(torch.cat((features[u], features[v])).unsqueeze(0))[0]:
            preprocessed_g.add_edge(u, v)
        else:
            K_neigbors = get_neigbors(original_g, v, depth=15)
            preprocessed_g.add_edge(u, get_one_homophilious_node(u, K_neigbors, model))

    return dgl.from_networkx(preprocessed_g)


if __name__ == "__main__":
    for data_name in datasets:
        print("*" * 20)
        print("baseline dataset:{}".format(data_name))
        g, nclass, features, labels, train, val, test = load_data(
            data_name, train_ratio
        )
        data = Graph(
            x=features,
            edge_index=torch.cat(
                (g.edges()[0].unsqueeze(1), g.edges()[1].unsqueeze(1)), dim=1
            ).T,
            y=labels,
            train_mask=train,
            val_mask=val,
            test_mask=test,
        )
        dataset = NodeDataset(data=data)
        experiment(
            task="node_classification",
            dataset=dataset,
            model=baseline_model,
            seed=list(range(10)),
        )

        print("*" * 20)
        print("rewire dataset:{}".format(data_name))
        predict_model = joblib.load(
            "./model/{}_{}.pkl".format(data_name, randint(0, 9))
        )
        print("befor:{}".format(int(g.number_of_edges() / 2)))
        g = processed_g_by_rewiring(g, predict_model)
        print("after:{}".format(int(g.number_of_edges() / 2)))
        data = Graph(
            x=features,
            edge_index=torch.cat(
                (g.edges()[0].unsqueeze(1), g.edges()[1].unsqueeze(1)), dim=1
            ).T,
            y=labels,
            train_mask=train,
            val_mask=val,
            test_mask=test,
        )
        dataset = NodeDataset(data=data)
        experiment(
            task="node_classification",
            dataset=dataset,
            model=baseline_model,
            seed=list(range(10)),
        )