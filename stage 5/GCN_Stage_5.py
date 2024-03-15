import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt
def set_seed(seed_value=10): #random number generation
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

set_seed(10)

class GraphConvolution(nn.Module):
    """Simple GCN layer"""
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)  #  second hidden layer
        self.gc3 = GraphConvolution(nhid2, nclass)  # Adjusted final layer
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))  # Pass through the second hidden layer
        x = F.dropout(x, self.dropout, training=self.training)  # Optional: additional dropout before the final layer
        x = self.gc3(x, adj)  # Output layer
        return F.log_softmax(x, dim=1)

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_encoded = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return np.argmax(labels_encoded, axis=1)

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj

def load_citeseer(node_path='/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/code/stage_5_code/stage_5_data/citeseer/node', edge_path='/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/code/stage_5_code/stage_5_data/citeseer/link'):
    # Load and process node features and labels
    node_data = np.genfromtxt(node_path, dtype=np.dtype(str))
    features = sp.csr_matrix(node_data[:, 1:-1], dtype=np.float32)
    labels = encode_labels(node_data[:, -1])

    # Map node IDs to indices
    node_id_to_idx = {j: i for i, j in enumerate(node_data[:, 0])}

    # Load and process edges
    edge_data = np.genfromtxt(edge_path, dtype=np.dtype(str))
    edges = np.array(list(map(lambda x: (node_id_to_idx[x[0]], node_id_to_idx[x[1]]), edge_data)))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_adj(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    return torch.FloatTensor(np.array(features.todense())), \
           torch.LongTensor(labels), \
           torch.FloatTensor(np.array(adj.todense()))

nfeat = 3703
nhid = 15     # First hidden layer size
nhid2 = 20    # Second hidden layer size
nclass = 6    # Number of classes
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 400

# Load Citeseer dataset
features, labels, adj = load_citeseer()

# Split dataset
idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.1, random_state=42, stratify=labels.numpy())

model = GCN(nfeat=nfeat, nhid=nhid, nhid2=nhid2, nclass=nclass, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item()

def test():
    model.eval()
    output = model(features, adj)
    pred = output[idx_test].max(1)[1].type_as(labels[idx_test])
    accuracy = pred.eq(labels[idx_test]).double().sum() / idx_test.shape[0]
    return accuracy.item()

accuracies = []
loss_values = []

for epoch in range(epochs):
    train_loss = train(epoch)
    loss = train(epoch)
    loss_values.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}')
    accuracies.append(test())

average_accuracy_percent = (sum(accuracies) / len(accuracies)) * 100
print(f'Accuracy for Citeseer dataset: {average_accuracy_percent:.2f}%')

plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs for Citeseer Dataset')
plt.legend()
plt.grid(True)
plt.show()

print("")
print("")


def set_seed(seed_value=10):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
set_seed(10)
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)  # New second hidden layer
        self.gc3 = GraphConvolution(nhid2, nclass)  # Adjusted final layer
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)  #additional dropout before the final layer
        x = self.gc3(x, adj)  # Output layer
        return F.log_softmax(x, dim=1)

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_encoded = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return np.argmax(labels_encoded, axis=1)

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj

def load_cora(node_path='/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/code/stage_5_code/stage_5_data/cora/node', edge_path='/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/code/stage_5_code/stage_5_data/cora/link'):
    # Load and process node features and labels
    node_data = np.genfromtxt(node_path, dtype=np.dtype(str))
    features = sp.csr_matrix(node_data[:, 1:-1], dtype=np.float32)
    labels = encode_labels(node_data[:, -1])

    # Map node IDs to indices
    node_id_to_idx = {j: i for i, j in enumerate(node_data[:, 0])}

    # Load and process edges
    edge_data = np.genfromtxt(edge_path, dtype=np.dtype(str))
    edges = np.array(list(map(lambda x: (node_id_to_idx[x[0]], node_id_to_idx[x[1]]), edge_data)))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_adj(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    return torch.FloatTensor(np.array(features.todense())), \
           torch.LongTensor(labels), \
           torch.FloatTensor(np.array(adj.todense()))

# Parameters
nfeat = 1433
nhid = 25
nhid2 = 25
nclass = 7  #Cora class labels
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 400

# Load Cora dataset
features, labels, adj = load_cora()

# Split dataset
idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.1, random_state=42, stratify=labels.numpy())

model = GCN(nfeat=nfeat, nhid=nhid, nhid2=nhid2, nclass=nclass, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item()

def test():
    model.eval()
    output = model(features, adj)
    pred = output[idx_test].max(1)[1].type_as(labels[idx_test])
    accuracy = pred.eq(labels[idx_test]).double().sum() / idx_test.shape[0]
    return accuracy.item()

accuracies = []
loss_values = []

for epoch in range(epochs):
    loss = train(epoch)
    loss_values.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')
    accuracies.append(test())  # Collect accuracy for each epoch

# After training, print the average test accuracy
average_accuracy_percent = (sum(accuracies) / len(accuracies)) * 100
print(f'Accuracy for Cora dataset: {average_accuracy_percent:.2f}%')

plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs for Cora Dataset')
plt.legend()
plt.grid(True)
plt.show()

print("")
print("")



def set_seed(seed_value=10):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

set_seed(10)
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)  # New second hidden layer
        self.gc3 = GraphConvolution(nhid2, nclass)  # Adjusted final layer
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)  # : additional dropout before the final layer
        x = self.gc3(x, adj)  # Output layer
        return F.log_softmax(x, dim=1)

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_encoded = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return np.argmax(labels_encoded, axis=1)

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj

def load_pubmed(node_path='/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/code/stage_5_code/stage_5_data/pubmed/node', edge_path='/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/code/stage_5_code/stage_5_data/pubmed/link'):
    # Load and process node features and labels
    node_data = np.genfromtxt(node_path, dtype=np.dtype(str))
    features = sp.csr_matrix(node_data[:, 1:-1], dtype=np.float32)
    labels = encode_labels(node_data[:, -1])

    node_id_to_idx = {j: i for i, j in enumerate(node_data[:, 0])}

    edge_data = np.genfromtxt(edge_path, dtype=np.dtype(str))
    edges = np.array(list(map(lambda x: (node_id_to_idx[x[0]], node_id_to_idx[x[1]]), edge_data)))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_adj(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    return torch.FloatTensor(np.array(features.todense())), \
           torch.LongTensor(labels), \
           torch.FloatTensor(np.array(adj.todense()))

# Parameters for PubMed
nfeat = 500
nhid = 16
nhid2 = 16
nclass = 3  # PubMed has 3 classes
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 200

# Load PubMed dataset
features, labels, adj = load_pubmed()

# Split dataset
idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.1, random_state=42, stratify=labels.numpy())

model = GCN(nfeat=nfeat, nhid=nhid, nhid2=nhid2, nclass=nclass, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item()

def test():
    model.eval()
    output = model(features, adj)
    pred = output[idx_test].max(1)[1].type_as(labels[idx_test])
    accuracy = pred.eq(labels[idx_test]).double().sum() / idx_test.shape[0]
    return accuracy.item()

accuracies = []
loss_values = []

for epoch in range(epochs):
    train_loss = train(epoch)
    loss = train(epoch)
    loss_values.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}')
    accuracies.append(test())

average_accuracy_percent = (sum(accuracies) / len(accuracies)) * 100
print(f'Accuracy for Pubmed dataset: {average_accuracy_percent:.2f}%')

plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs for Pubmed Dataset')
plt.legend()
plt.grid(True)
plt.show()