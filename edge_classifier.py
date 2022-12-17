import joblib
from utils import load_data, set_seed
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class HelpDataset(Dataset):
    def __init__(self, features, labels, indexes):
        self.features = features
        self.labels = labels
        self.indexes = indexes

    def __getitem__(self, index):
        (index1, index2) = divmod(self.indexes[index].item(), len(self.labels))
        return (
            torch.cat((self.features[index1], self.features[index2]), -1),
            (self.labels[index1] == self.labels[index2]).long(),
        )

    def __len__(self):
        return len(self.indexes)


def get_index_from_g(g, labels):
    return_index = []
    source, traget = g.edges()
    for i in range(g.num_edges()):
        return_index.append(source[i].item() * labels.shape[0] + traget[i].item())
    return return_index


datasets = ["cornell", "texas", "wisconsin", "chameleon"]
train_ratio = 0.6
batch_size = 1024 * 64

for data_name in datasets:
    print("*" * 20)
    print("dataset:{}".format(data_name))

    g, nclass, features, labels, train, val, test = load_data(data_name, train_ratio)
    avail_index = torch.cat((train, val))
    train_index = torch.tensor(range(avail_index.shape[0] ** 2))
    data_train = HelpDataset(features, labels, train_index)
    test_index = torch.tensor(get_index_from_g(g, labels))
    print("The number of edges:{}".format(test_index.shape[0]))
    data_test = HelpDataset(features, labels, test_index)

    results = []
    for i in range(10):
        dataload_train = DataLoader(
            dataset=data_train, batch_size=batch_size, shuffle=True
        )
        dataload_test = DataLoader(dataset=data_test, batch_size=batch_size)
        model = MLPClassifier()
        set_seed(i)  # set random seed

        batch_time = 0
        for data in dataload_train:
            inputs, outputs = data
            model.fit(inputs, outputs)
            batch_time += 1
        print("batch times:{}".format(batch_time))
        # save model
        joblib.dump(model, "./model/{}_{}.pkl".format(data_name, i))

        # load model
        model = joblib.load("./model/{}_{}.pkl".format(data_name, i))
        result = []
        for data in dataload_test:  #  dataload_test  , helper_dataload_test
            inputs, outputs = data
            y_pred = model.predict(inputs)
            result.append(accuracy_score(y_pred, outputs, normalize=False))
        results.append(np.array(result).sum() / test_index.shape[0])
    print(
        "mean~std:{}~{}".format(
            np.round(np.array(results).mean(), 4), np.round(np.array(results).std(), 4)
        )
    )
    print([round(re, 4) for re in results])
