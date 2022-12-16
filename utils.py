import numpy as np
import scipy.sparse as sp
import torch
import random
import networkx as nx
import dgl


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_data(dataset, train_ratio):

    if dataset in ["cora", "citeseer", "pubmed"]:

        edge = np.loadtxt("./data/low_freq/{}.edge".format(dataset), dtype=int).tolist()
        feat = np.loadtxt("./data/low_freq/{}.feature".format(dataset))
        labels = np.loadtxt("./data/low_freq/{}.label".format(dataset), dtype=int)
        train = np.loadtxt("./data/low_freq/{}.train".format(dataset), dtype=int)
        val = np.loadtxt("./data/low_freq/{}.val".format(dataset), dtype=int)
        test = np.loadtxt("./data/low_freq/{}.test".format(dataset), dtype=int)
        nclass = len(set(labels.tolist()))
        print(dataset, nclass)

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        feat = normalize_features(feat)
        feat = torch.FloatTensor(feat)
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)

        return g, nclass, feat, labels, train, val, test

    elif dataset in ["film"]:
        graph_adjacency_list_file_path = (
            "./data/high_freq/{}/out1_graph_edges.txt".format(dataset)
        )
        graph_node_features_and_labels_file_path = (
            "./data/high_freq/{}/out1_node_feature_label.txt".format(dataset)
        )

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset == "film":
            with open(
                graph_node_features_and_labels_file_path
            ) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split("\t")
                    assert len(line) == 3
                    assert (
                        int(line[0]) not in graph_node_features_dict
                        and int(line[0]) not in graph_labels_dict
                    )
                    feature_blank = np.zeros(932, dtype=np.uint16)
                    feature_blank[np.array(line[1].split(","), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(
                graph_node_features_and_labels_file_path
            ) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split("\t")
                    assert len(line) == 3
                    assert (
                        int(line[0]) not in graph_node_features_dict
                        and int(line[0]) not in graph_labels_dict
                    )
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(","), dtype=np.uint8
                    )
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split("\t")
                assert len(line) == 2
                if int(line[0]) not in G:
                    G.add_node(
                        int(line[0]),
                        features=graph_node_features_dict[int(line[0])],
                        label=graph_labels_dict[int(line[0])],
                    )
                if int(line[1]) not in G:
                    G.add_node(
                        int(line[1]),
                        features=graph_node_features_dict[int(line[1])],
                        label=graph_labels_dict[int(line[1])],
                    )
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        row, col = np.where(adj.todense() > 0)

        U = row.tolist()
        V = col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        features = np.array(
            [
                features
                for _, features in sorted(G.nodes(data="features"), key=lambda x: x[0])
            ],
            dtype=float,
        )
        labels = np.array(
            [label for _, label in sorted(G.nodes(data="label"), key=lambda x: x[0])],
            dtype=int,
        )

        n = labels.shape[0]
        idx = [i for i in range(n)]
        # random.shuffle(idx)
        r0 = int(n * train_ratio)
        r1 = int(n * 0.6)
        r2 = int(n * 0.8)

        idx_train = np.array(idx[:r0])
        idx_val = np.array(idx[r1:r2])
        idx_test = np.array(idx[r2:])

        features = normalize_features(features)
        features = torch.FloatTensor(features)

        nclass = 5
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(idx_train)
        val = torch.LongTensor(idx_val)
        test = torch.LongTensor(idx_test)
        print(dataset, nclass)

        return g, nclass, features, labels, train, val, test

    # datasets in Geom-GCN
    elif dataset in ["cornell", "texas", "wisconsin", "chameleon", "squirrel"]:

        graph_adjacency_list_file_path = (
            "./data/high_freq/{}/out1_graph_edges.txt".format(dataset)
        )
        graph_node_features_and_labels_file_path = (
            "./data/high_freq/{}/out1_node_feature_label.txt".format(dataset)
        )

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        with open(
            graph_node_features_and_labels_file_path
        ) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split("\t")
                assert len(line) == 3
                assert (
                    int(line[0]) not in graph_node_features_dict
                    and int(line[0]) not in graph_labels_dict
                )
                graph_node_features_dict[int(line[0])] = np.array(
                    line[1].split(","), dtype=np.uint8
                )
                graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split("\t")
                assert len(line) == 2
                if int(line[0]) not in G:
                    G.add_node(
                        int(line[0]),
                        features=graph_node_features_dict[int(line[0])],
                        label=graph_labels_dict[int(line[0])],
                    )
                if int(line[1]) not in G:
                    G.add_node(
                        int(line[1]),
                        features=graph_node_features_dict[int(line[1])],
                        label=graph_labels_dict[int(line[1])],
                    )
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [
                features
                for _, features in sorted(G.nodes(data="features"), key=lambda x: x[0])
            ]
        )
        labels = np.array(
            [label for _, label in sorted(G.nodes(data="label"), key=lambda x: x[0])]
        )

        features = normalize_features(features)

        # g = DGLGraph(adj)
        g = dgl.from_scipy(adj)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        n = len(labels.tolist())
        idx = [i for i in range(n)]
        # random.shuffle(idx)
        r0 = int(n * train_ratio)
        r1 = int(n * 0.6)
        r2 = int(n * 0.8)
        train = np.array(idx[:r0])
        val = np.array(idx[r1:r2])
        test = np.array(idx[r2:])

        nclass = len(set(labels.tolist()))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)
        print(dataset, nclass)

        return g, nclass, features, labels, train, val, test


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype(np.float64)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)