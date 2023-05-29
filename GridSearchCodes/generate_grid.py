import os
import numpy as np
import pandas as pd
#import logging
import time
from itertools import product

import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score

from gridsearch_GeNNius import GridSearchModel, EarlyStopper, shuffle_label_data

print('-')

def return_values(DATABASE='drugbank', hidden_channels=7, learning_rate=0.01, nb_hidden_layer=2, layer_type='sage', act='relu', aggr_type='sum', n_heads=1, p=0.2):
    

    list_dict = ['param_database', 'param_hidden_channels', 
                'param_learning_rate', 'param_nb_hidden_layer', 
                'param_layer_type', 'param_act', 
                'param_aggr_type', 'param_n_heads',
                'param_p', 
                'auc_train', 'auc_val', 'auc_test',
                'aupr_train', 'aupr_val', 'aupr_test',
                'iter', 'time']

    dict_values = dict(zip(list_dict, [None]*len(list_dict)))
    dict_values['param_database'] = DATABASE
    dict_values['param_hidden_channels'] = hidden_channels
    dict_values['param_learning_rate'] = learning_rate
    dict_values['param_nb_hidden_layer'] = nb_hidden_layer
    dict_values['param_layer_type'] = layer_type
    dict_values['param_act'] = act
    dict_values['param_p'] =  p

    if layer_type in ['sage', 'graphconv']:
        n_heads = np.nan
    elif layer_type in ['transformerconv', 'gatconv']:
        aggr_type = np.nan
    
    dict_values['param_aggr_type'] = aggr_type
    dict_values['param_n_heads'] = n_heads

    def train(train_data):

        model.train()
        optimizer.zero_grad()
        
        train_data = shuffle_label_data(train_data)
        
        _, pred = model(train_data.x_dict, train_data.edge_index_dict,
                    train_data['drug', 'protein'].edge_label_index)

        target = train_data['drug', 'protein'].edge_label
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        return float(loss)


    @torch.no_grad()
    def test(data):
        model.eval()
        emb, pred = model(data.x_dict, data.edge_index_dict,
                    data['drug', 'protein'].edge_label_index)

        # target value
        target = data['drug', 'protein'].edge_label.float()
        
        # loss
        loss = criterion(pred, target)

        # auroc
        out = pred.view(-1).sigmoid()

        # calculate metrics
        auc = roc_auc_score(target.cpu().numpy(), out.detach().cpu().numpy())
        aupr = average_precision_score(target.cpu().numpy(), out.detach().cpu().numpy())

        return round(auc, 6), emb, out, loss.cpu().numpy(), aupr

    #########

    DATABASE = DATABASE.lower()

    print('Dataset: ', DATABASE)

    #PATH_DATA = os.path.join('Data', DATABASE.upper(), f'hetero_data_{DATABASE}.pt')
    PATH_DATA = os.path.join('../GENNIUS/Data', DATABASE.upper(), f'hetero_data_{DATABASE}.pt')

    print('reading from', PATH_DATA)


    print('hd: ', hidden_channels)
    
    # Load data
    data = torch.load(PATH_DATA)
    #logging.debug(f'Data is cuda?: {data.is_cuda}')

    # Prepare data
    data = T.ToUndirected()(data)
    # Remove "reverse" label.
    del data['protein', 'rev_interaction', 'drug'].edge_label  
    
    split = T.RandomLinkSplit(
        num_val= 0.1,
        num_test= 0.2, 
        is_undirected= True,
        add_negative_train_samples= True, # False for: Not adding negative links to train
        neg_sampling_ratio= 1.0, # ratio of negative sampling is 0
        disjoint_train_ratio = 0.2, #
        edge_types=[('drug', 'interaction', 'protein')],
        rev_edge_types=[('protein', 'rev_interaction', 'drug')],
        split_labels=False
    )

    train_data, val_data, test_data = split(data)

    # logging.debug(f"Number of nodes\ntrain: {train_data.num_nodes}\ntest: {test_data.num_nodes}\nval: {val_data.num_nodes}")
    # logging.debug(f"Number of edges (label)\ntrain: {train_data['drug', 'protein'].edge_label.size()}")
    # logging.debug(f"test: {test_data['drug', 'protein'].edge_label.size()}")
    # logging.debug(f"val: {val_data['drug', 'protein'].edge_label.size()}")



    ## Run model
    device = 'cuda'

    #logging.info(f'hidden channels: {hidden_channels}')

    model = GridSearchModel(hidden_channels=hidden_channels, data=data, nb_hidden_layer=nb_hidden_layer, layer_type=layer_type, act=act, aggr_type=aggr_type, n_heads=n_heads, p=p).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 0.01
    criterion = torch.nn.BCEWithLogitsLoss()

    early_stopper = EarlyStopper(tolerance=10, min_delta=0.05)

    # lazy init, ned to rune one mode step for inferring number of parameters 
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)


    init_time = time.time()

    for epoch in range(1_000): 
        loss = train(train_data)
        train_auc, _, _, train_loss, train_aupr = test(train_data)
        val_auc, _, _, val_loss, val_aupr = test(val_data)
        test_auc, emb, _, test_loss, test_aupr= test(test_data)
        if epoch%50 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_auc:.4f}, '
                    f'Val: {val_auc:.4f}, Test: {test_auc:.4f}')

        if early_stopper.early_stop(train_loss, val_loss) and epoch>40:    
            print('early stopped at epoch ', epoch)         
            break
    
    end_time = time.time()

    print(" ")
    print(f"Final AUC Train: {train_auc:.4f}, AUC Val {val_auc:.4f},AUC Test: {test_auc:.4f}")
    print(f"Final AUPR Train: {train_aupr:.4f}, AUC Val {val_aupr:.4f},AUC Test: {test_aupr:.4f}")
    elap_time = (end_time-init_time)/60
    print(f"Elapsed time {elap_time:.4f} min")
    
    dict_values['auc_train'] = train_auc
    dict_values['auc_val'] = val_auc
    dict_values['auc_test'] = test_auc

    dict_values['aupr_train'] = train_aupr
    dict_values['aupr_val'] = val_aupr
    dict_values['aupr_test'] = test_aupr

    dict_values['iter'] = epoch
    dict_values['time'] = elap_time

    return dict_values



def exceptions_prints(layer, act, hd, n_layers, lr, p, aggr, n_heads):
    print("******** Something went wrong with params: ********\n")
    print(f"layer: {layer}")
    print(f"act: {act}")
    print(f"hd: {hd}")
    print(f"n_layers: {n_layers}")
    print(f"lr: {lr}")
    print(f"p: {p}")
    print(f"aggr: {aggr}")
    print(f"n_heads: {n_heads}")
    print("********\n")

##############################################################
###################### LAUNCH ################################


list_dict = ['param_database', 'param_hidden_channels', 
            'param_learning_rate', 'param_nb_hidden_layer', 
            'param_layer_type', 'param_act', 
            'param_aggr_type', 'param_n_heads',
            'param_p', 
            'auc_train', 'auc_val', 'auc_test',
            'aupr_train', 'aupr_val', 'aupr_test',
            'iter', 'time']

cv_results_ = pd.DataFrame(columns=list_dict)



layer_types = ['sage', 'graphconv', 'gatconv', 'transformerconv'] 
act_types = ['relu', 'tanh'] # 

n_hiddenlayers_list = [1, 2, 3, 4, 5] 

lr_rates = [0.01]
drop_rates = [0, 0.2, 0.5]

aggregation_types = ['sum', 'mean', 'max' ] # only for layers that need aggregators
list_n_heads = [1, 2, 4]  # only for layers with heads

hd_list =  [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 40, 50, 100]  

drop_rates = [0, 0.2, 0.5] #

OUTPUT_TSV = 'Results/cv_results_embeddingdim_ext_10.tsv' #



nreps = 10  #
print(f'Doing {nreps} repetitions')
i=0

for rep in range(nreps):
    for layer, act, hd, n_layers, aggr, n_heads, lr, p in product(layer_types, act_types, hd_list, n_hiddenlayers_list, aggregation_types, list_n_heads, lr_rates, drop_rates):
    #for hd, p in product(hd_list, drop_rates):
        #for aggr in aggregation_types:
        i+=1
        res_dict = {}

        print("==========================\n")
        print(rep, layer, act, hd, n_layers, lr, p, aggr, n_heads) 

        try:
            res_dict = return_values(DATABASE='drugbank', hidden_channels=hd, learning_rate=lr, nb_hidden_layer=n_layers, layer_type=layer, act=act, aggr_type=aggr, n_heads=n_heads, p=p)
            cv_results_ = cv_results_.append(res_dict, ignore_index=True)
            cv_results_.to_csv(OUTPUT_TSV, sep="\t", index=False)

        except Exception as e: 
            print(e)
            exceptions_prints(layer, act, hd, n_layers, lr, p, aggr, n_heads)
    

print(cv_results_)

tt = 0.3
print(i)
print(i*tt, 'min')
print(i*tt/60, 'h')
print(i*tt/60/24, 'day')



