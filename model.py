import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.n_class = n_class
        self.model = nn.Sequential(
            nn.Linear(10, 8),
            nn.Linear(8, 7),
            nn.Linear(7, n_class)
        )

    def remove_class(self, remove_idx):
        new_model = nn.Sequential(
            nn.Linear(10, 8),
            nn.Linear(8, 7),
            nn.Linear(7, self.n_class - 1)
        )

        for idx in range(len(self.model) - 1):
            new_model[idx].weight = nn.Parameter(self.model[idx].weight)
            new_model[idx].bias = nn.Parameter(self.model[idx].bias)

        old_weight = self.model[-1].weight
        new_weight = new_model[-1].weight
        old_bias = self.model[-1].bias
        new_bias = new_model[-1].bias

        for i in range(old_weight.shape[0]):
            pos = 0
            for j in range(old_weight.shape[1]):
                if i != remove_idx:
                    with torch.no_grad():
                        new_weight[i][pos] = old_weight[i][j]
                        pos += 1

        pos = 0
        for i in range(old_bias.shape[0]):
            if i != remove_idx:
                with torch.no_grad():
                    new_bias[pos] = old_bias[i]
                    pos += 1

        self.model[-1].weight = nn.Parameter(new_weight)
        self.model[-1].bias = nn.Parameter(new_bias)
        self.model = new_model
        self.n_class = self.n_class - 1

    def add_class(self):
        new_model = nn.Sequential(
            nn.Linear(10, 8),
            nn.Linear(8, 7),
            nn.Linear(7, self.n_class + 1)
        )

        for idx in range(len(self.model) - 1):
            new_model[idx].weight = nn.Parameter(self.model[idx].weight)
            new_model[idx].bias = nn.Parameter(self.model[idx].bias)

        old_weight = self.model[-1].weight
        new_weight = new_model[-1].weight
        old_bias = self.model[-1].bias
        new_bias = new_model[-1].bias

        for i in range(old_weight.shape[0]):
            for j in range(old_weight.shape[1]):
                with torch.no_grad():
                    new_weight[i][j] = old_weight[i][j]
        for i in range(old_bias.shape[0]):
            with torch.no_grad():
                new_bias[i] = old_bias[i]

        self.model[-1].weight = nn.Parameter(new_weight)
        self.model[-1].bias = nn.Parameter(new_bias)
        self.model = new_model
        self.n_class = self.n_class + 1


class Node:
    def __init__(self, label, value, nodes, class_pos):
        self.label = label
        self.value = value
        self.nodes = nodes
        self.class_pos = class_pos


class NetNode:
    def __init__(self, n_class, labels=None, max_leaf=10):
        if labels is None:
            labels = []
        self.net = Net(n_class)
        self.n_class = n_class
        self.child = [label for label in labels]
        self.child_leaf = [True for _ in range(n_class)]
        self.labels = labels
        self.max_leaf = max_leaf
        self.labels_transform = {}

        for cls in range(n_class):
            self.labels_transform[labels[cls]] = []
            self.labels_transform[labels[cls]].append(labels[cls])

    def get_num_leaf_node(self):
        cnt = 0
        for is_leaf in self.child_leaf:
            if is_leaf:
                cnt += 1
        return cnt

    def remove_leaf(self, label):
        child = []
        child_leaf = []
        labels = []
        self.n_class = (self.n_class - 1)
        position_in_net = -1

        for i in range(len(self.labels)):
            if self.labels[i] != label:
                child.append(self.child[i])
                child_leaf.append(self.child_leaf[i])
                labels.append(self.labels[i])
            else:
                position_in_net = i

        self.child = child
        self.child_leaf = child_leaf
        self.labels = labels
        self.net.remove_class(position_in_net)

    def add_leaf(self, label):
        self.child.append(label)
        self.child_leaf.append(True)
        self.n_class = (self.n_class + 1)
        self.labels.append(label)
        self.labels_transform[label] = []
        self.labels_transform[label].append(label)
        self.net.add_class()


class TreeCNN:
    def __init__(self, init_label, alpha=0.1, beta=0.1, max_leaf_node=100):
        n_class_init = len(init_label)
        self.root = NetNode(n_class_init, init_label)
        self.alpha = alpha
        self.beta = beta
        self.max_leaf_node = max_leaf_node

    def add_task(self, label=[]):
        self.grow_net(self.root, label)

    def grow_net(self, root, data_of_cls=[], labels=[]):
        def get_avg_mat(root, data_of_cls_, labels_):
            avg = torch.zeros((root.n_class, 0))
            for x, y in zip(data_of_cls_, labels_):
                out = root(x)
                avg_i = torch.mean(out, dim=0)
                avg = torch.concat((avg_i, avg_i.reshape(avg_i.shape[0], 1)), dim=1)
            return avg

        def get_loglikelihood(avg):
            return torch.pow(torch.e, avg) / torch.sum(torch.pow(torch.e, avg), dim=0)

        def gen_list(lh, labels_in):
            lists = []
            for i in range(lh.shape[1]):
                label = labels_in[i]
                values = []
                nodes = []
                col = lh[:, i].copy()
                for _ in range(3):
                    max_idx = torch.argmax(col)
                    values.append(col[max_idx])
                    nodes.append(max_idx)
                    col[max_idx] = -100

                lists.append(Node(label, values, nodes, i))
            lists.sort(key=lambda node_list: node_list.values[0])

            return lists

        llh = get_loglikelihood(get_avg_mat(root, data_of_cls, labels))

        print(llh)


if __name__ == "__main__":
    net = TreeCNN([1, 2, 3])
    net.add_task()

    model = Net(5)
    print(model)
    model.remove_class(4)
    print(model)
    model.add_class()
    print(model)
