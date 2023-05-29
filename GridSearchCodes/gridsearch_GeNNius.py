import numpy as np
import torch
from torch_geometric.nn import to_hetero, SAGEConv, GraphConv, TransformerConv, GATConv

from torch.nn import Linear, Dropout, ReLU, Tanh
import random




class GrigSearchGNNEncoder(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels, nb_hidden_layer=2, layer_type='sage', act='relu', aggr_type='sum', n_heads=1, p=0.2):
        
        super().__init__()

        self.dropout = Dropout(p)
        self.nb_hl = nb_hidden_layer
        layer_str2func = {'sage': SAGEConv,
                          'graphconv': GraphConv,
                          'transformerconv': TransformerConv, #change layers and parameters heads below
                          'gatconv': GATConv # change layers and parameters heads below

                          }
        act_str2func = {'relu': ReLU(),
                        'tanh': Tanh()
                        }
        
        self.layer = layer_str2func.get(layer_type.lower())
        self.act = act_str2func.get(act.lower()) #act() # activation function

        self.aggr = aggr_type 
        self.heads = n_heads

        if layer_type in ['sage', 'graphconv']:
            #print('here sage etc')
            self.heads = 'Not using heads'
            self.conv_in = self.layer((-1, -1), hidden_channels, aggr=self.aggr)
            self.conv_med = self.layer((-1, -1), hidden_channels, aggr=self.aggr)
            self.conv_out = self.layer((-1, -1), out_channels, aggr=self.aggr)

        elif layer_type in ['transformerconv', 'gatconv']:
            #print('gats')
            print(f'heads {self.heads}')
            self.aggr = 'Not using aggregation'
            self.conv_in = self.layer((-1, -1), hidden_channels, add_self_loops=False, n_heads=self.heads)
            self.conv_med = self.layer((-1, -1), hidden_channels,  add_self_loops=False, n_heads=self.heads)
            self.conv_out = self.layer((-1, -1), out_channels, add_self_loops=False, n_heads=self.heads)

        else:
            raise NotImplementedError
    


    def forward(self, x, edge_index):
        print(f'Using: {self.layer} with act {self.act} and aggregation {self.aggr} // heads {self.heads}')
        if self.nb_hl == 1:
            # one layer special case
            x = self.conv_out(x, edge_index) # direcly ouput dimension
            return x
        
        elif self.nb_hl == 2:
            x = self.conv_in(x, edge_index) # direcly ouput dimension
            x = self.act(x) # apply activation
            x = self.dropout(x) # apply dropout
            x = self.conv_out(x, edge_index) # direcly ouput dimension
            return x
        
        else:
            x = self.conv_in(x, edge_index) # direcly ouput dimension
            print('init layer 0 : ', x)
            x = self.act(x) # apply activation
            x = self.dropout(x) # apply dropout
            for i in range(self.nb_hl-2):
                x = self.conv_med(x, edge_index) # direcly ouput dimension
                print('med layer', i+1, x)
                x = self.act(x) # apply activation
                x = self.dropout(x) # apply dropout
            x = self.conv_out(x, edge_index) # direcly ouput dimension
            print(f'final layer: {self.nb_hl-2+1}', x)
            return x




################################################################################
################################################################################




################################################################################
################################################################################


class EdgeClassifier(torch.nn.Module):

    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
    
    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['drug'][row], z_dict['protein'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)





class GridSearchModel(torch.nn.Module):

    def __init__(self, hidden_channels, data,
                 nb_hidden_layer=2, layer_type='sage', act='relu', aggr_type='sum',n_heads=1, p=0.2):
        super().__init__()
        self.encoder = GrigSearchGNNEncoder(hidden_channels, hidden_channels, nb_hidden_layer, layer_type, act, aggr_type,n_heads, p)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeClassifier(hidden_channels)
    
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        out = self.decoder(z_dict, edge_label_index)
        return z_dict, out




class EarlyStopper():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0

    def early_stop(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                return True
        return False



def shuffle_label_data(train_data, a = ('drug', 'interaction', 'protein') ):
    
    length = train_data[a].edge_label.shape[0]
    lff = list(np.arange(length))
    random.shuffle(lff)

    train_data[a].edge_label = train_data[a].edge_label[lff]
    train_data[a].edge_label_index =  train_data[a].edge_label_index[:, lff]
    return train_data