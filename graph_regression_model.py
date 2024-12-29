import pickle

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, GATConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing

from glob import glob
import seaborn as sns

class GATv2(torch.nn.Module):
    def __init__(self, n_features, dim_h, out_dim):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(n_features, dim_h, edge_dim = 1)
        self.conv2 = GATv2Conv(dim_h, dim_h, edge_dim = 1)
        self.conv3 = GATv2Conv(dim_h, dim_h, edge_dim = 1)
        self.lin = Linear(dim_h, out_dim)
    
    def embed(self, x, edge_index, edge_attr, batch):
        # Node embeddings 
        h = self.conv1(x, edge_index, edge_attr= edge_attr)
        h = h.relu()
        h = self.conv2(h, edge_index, edge_attr= edge_attr)
        h = h.relu()
        h = self.conv3(h, edge_index, edge_attr= edge_attr)

        # Graph-level readout
        return global_mean_pool(h, batch)

    def forward(self, x, edge_index, edge_attr, batch):
        # Node embeddings 
        #h = self.conv1(x, edge_index, edge_attr= edge_attr)
        #h = h.relu()
        #h = self.conv2(h, edge_index, edge_attr= edge_attr)
        #h = h.relu()
        #h = self.conv3(h, edge_index, edge_attr= edge_attr)

        # Graph-level readout
        #hG = global_mean_pool(h, batch)
        hG = self.embed(x, edge_index, edge_attr= edge_attr,batch=batch)

        # Classifier
        #h = F.dropout(hG, p=0.1, training=self.training)
        h = self.lin(hG)
        
        return h

def train(model,epochs, loader, val_loader, save_path):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    iteration = 0
    
    update_iter = 500
    
    model.train()
    running_loss = 0
    for epoch in range(epochs+1):
        
        # Train on batches
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(100*out, 100*data.mrmse)
            running_loss += loss.item()
            
            
            loss.backward()
            optimizer.step()

            iteration += 1
            print(iteration, end='\r')

            if iteration % update_iter == 0:
                # Validation
                val_loss = test(model, val_loader)
                val_losses.append(val_loss)
                # Print metrics every 20 epochs
                print(f'Epoch {epoch:>3} | Train Loss: {running_loss/update_iter:} | Val Loss: {val_loss:}', end='\r' )
                train_losses.append(running_loss/update_iter)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, save_path) 
                running_loss = 0
                
            
    return model, train_losses, val_losses

@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.MSELoss()
    model.eval()
    loss = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss += criterion(100*out, 100*data.mrmse) 

    return loss / len(loader)

storage = dict()
file_paths = glob('reservoir_data/graph_regression/data_list_*.pkl')

for file_path in file_paths:
    with open(file_path,'rb') as f:
        data = pickle.load(f)

    import random
    from sklearn.model_selection import train_test_split

    train_set, test_set = train_test_split(data, test_size = 0.2)
    val_set, test_set = train_test_split(test_set, test_size = 0.5)

    # we need to store this
    train_len = len(train_set)
    val_len = len(val_set)
    test_len = len(test_set)

    train_loader = DataLoader(train_set, batch_size=32,num_workers=3)
    val_loader = DataLoader(val_set, batch_size=64,num_workers=3)
    test_loader = DataLoader(test_set, batch_size=64,num_workers=3)

    device = 'cpu'
    n_features = 125
    dim_h = 64
    out_dim = 1
    epochs = 100

    gcn = GATv2(n_features=n_features, dim_h=dim_h, out_dim=out_dim)
    gcn = gcn.to(device)

    name = file_path.split('/')[-1].split('_',2)[-1].split('.')[0]
    save_path = 'reservoir_data/graph_regression/' + name + '_best_model.pth'
    print(save_path)
    

    # we need to store val_losses and train_losses
    _, train_losses, val_losses = train(gcn, epochs, train_loader, val_loader, save_path)
    
    storage[name] = {"train_len":train_len,
                     "val_len":val_len,
                     "test_len":test_len,
                     "train_losses":train_losses,
                     "val_losses":val_losses}

storage_path = 'reservoir_data/graph_regression/graph_regression_result_info.pkl'
with open(storage_path, 'wb') as f:
    pickle.dump(storage, f)
