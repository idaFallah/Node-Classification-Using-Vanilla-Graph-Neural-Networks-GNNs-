
!pip install -q torch-scatter   torch-sparse   torch-cluster   torch-spline-conv   torch-geometric

# import libs

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import FacebookPagePage

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The Cora dataset

# import dataset from pytorch geometric
dataset = Planetoid(root='.', name='Cora')

data = dataset[0]

# info of dataset
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of nodes: {data.x.shape[0]}')


# info of graph
print(f'\nGraph:')
print('======================')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')

dataset.data

dataset.x

dataset.y

# seeing wehre all the edges are present

dataset.edge_index

dataset.edge_index.shape

# the Facebook dataset(if there's connection between two pages in facebook)

dataset = FacebookPagePage(root='.')

data = dataset[0]

# info of dataset
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of nodes: {data.x.shape[0]}')


# info of graph
print(f'\nGraph:')
print('======================')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')

# create masks

data.train_mask = range(18000)
data.val_mask = range(18001, 20000)
data.test_mask = range(20001, 22470)

dataset.data

dataset.x

dataset.y

dataset.edge_index

# creating a vanilla NN

# import libs

import pandas as pd
from torch.nn import Linear
import torch.nn.functional as F

torch.manual_seed(0)

dataset = Planetoid(root='.', name='Cora')
data = dataset[0]

dataset

# creating a dataframe from x features

dataframe_x = pd.DataFrame(data.x.numpy())
dataframe_x['label'] = pd.DataFrame(data.y)
dataframe_x

# creating a func. for calculating the accuracy of the model

def accuracy(y_pred, y_true):
  return torch.sum(y_pred == y_true) / len(y_true)

# class of Multilayer Preceptron
class MLP(torch.nn.Module):
  def __init__(self, dim_in, dim_h, dim_out):  # input dimension, hidden layer dimesnion, ouput dimension
    super().__init__()
    self.linear1 = Linear(dim_in, dim_h)
    self.linear2 = Linear(dim_h, dim_out)

  def forward(self, x):
    x = self.linear1(x)
    x = torch.relu(x)
    x = self.linear2(x)
    return F.log_softmax(x, dim=1)

  def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.parameters(),
                                 lr=0.01,
                                 weight_decay=5e-4)
    self.train()
    for epoch in range(epochs+1):
      optimizer.zero_grad()
      out = self(data.x)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      acc = accuracy(out[data.train_mask].argmax(dim=1),
                     data.y[data.train_mask])
      loss.backward()
      optimizer.step()

      if(epoch % 20 == 0):
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                           data.y[data.val_mask])
        print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train acc:'
        f' {acc*100:>5.2f}% | Val loss: {val_loss:.2f} | '
        f'Val acc: {val_acc*100:.2f}%')


  @torch.no_grad()
  def test(self, data):
    self.eval()
    out = self(data.x)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

mlp = MLP(dataset.num_features, 16, dataset.num_classes)
mlp

# train

mlp.fit(data, epochs=100)

acc = mlp.test(data)
print(f'\nMLP test accuracy: {acc*100:.2f}%')

# a sinlge layer of vanilla GNN
class vanillaGNNLayer(torch.nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.linear = Linear(dim_in, dim_out, bias=False)

  def forward(self, x, adjacency):
    x = self.linear(x)
    x = torch.sparse.mm(adjacency, x) # matrix multiplation of x(input matrix) and adjacency matrix so that we have the info of the central node in it too
    return x

from torch_geometric.utils import to_dense_adj # this method creates adjacency matrix

adjacency = to_dense_adj(data.edge_index)[0]
adjacency += torch.eye(len(adjacency))  # creating an identity matrix and adding it to adjacency matrix(which contains binary values of 0 & 1)
adjacency

class VanillaGNN(torch.nn.Module):
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.gnn1 = vanillaGNNLayer(dim_in, dim_h)
    self.gnn2 = vanillaGNNLayer(dim_h, dim_out)

  def forward(self, x, adjacency):
    h = self.gnn1(x, adjacency)
    h = torch.relu(h)
    h = self.gnn2(h, adjacency)
    return F.log_softmax(h, dim=1)

  def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.parameters(),
                                 lr=0.01,
                                 weight_decay=5e-4)
    self.train()
    for epoch in range(epochs+1):
      optimizer.zero_grad()
      out = self(data.x, adjacency)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      acc = accuracy(out[data.train_mask].argmax(dim=1),
                     data.y[data.train_mask])
      loss.backward()
      optimizer.step()

      if(epoch % 20 == 0):
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                           data.y[data.val_mask])

        print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train acc:'
        f' {acc*100:>5.2f}% | Val loss: {val_loss:.2f} | '
        f'Val acc: {val_acc*100:.2f}%')

  @torch.no_grad()
  def test(self, data):
    self.eval()
    out = self(data.x, adjacency)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

def accuracy(y_pred, y_true):
  return torch.sum(y_pred == y_true) / len(y_true)

# create the vanilla GNN model
gnn = VanillaGNN(dataset.num_features, 16, dataset.num_classes)
print(gnn)

# train the model
gnn.fit(data, epochs=100)

# evaluating the gnn model
acc = gnn.test(data)
print(f'\nGNN test accuracy: {acc*100:.2f}%')

# multiplying the output of linear layer with adjacency matrix has improved the model
# this action enables us to use the information of neighbor nodes as well the node itself





