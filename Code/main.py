import os
import numpy as np
import logging
import argparse
import time

import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score

from GeNNius import Model, EarlyStopper, shuffle_label_data
from utils import plot_auc



######################################## START MAIN #########################################
#############################################################################################
def main():
    # Get database parameter from user

    parser = argparse.ArgumentParser() 
    parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=3,
                    help="Verbosity (between 1-4 occurrences with more leading to more "
                        "verbose logging). CRITICAL=0, ERROR=1, WARN=2, INFO=3, "
                        "DEBUG=4")
    parser.add_argument("-d", "--database", help="database: e, nr, ic, gpcr, drugbank", type=str)
    parser.add_argument("-e", "--hidden", default=17,help="specify dimension of embedding", type=int)

    args = parser.parse_args()
    log_levels = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
    }
    # set the logging info
    level= log_levels[args.verbosity]
    fmt = '[%(levelname)s] %(message)s]'
    logging.basicConfig(format=fmt, level=level)

    #########

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

    DATABASE = args.database.lower()
    print('Dataset: ', DATABASE)

    PATH_DATA = os.path.join('Data', DATABASE.upper(), f'hetero_data_{DATABASE}.pt')
    print('reading from', PATH_DATA)


    hidden_channels = int(args.hidden)
    print('hd: ', hidden_channels)

    # device
    device = 'cuda'
    #device = 'cpu'
    print(device)
    
    # Load data
    data = torch.load(PATH_DATA)
    logging.debug(f'Data is cuda?: {data.is_cuda}')
    print(f'Data is cuda?: {data.is_cuda}')
    #data.to(device)
    #print(f'Data is cuda?: {data.is_cuda}')

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

    logging.debug(f"Number of nodes\ntrain: {train_data.num_nodes}\ntest: {test_data.num_nodes}\nval: {val_data.num_nodes}")
    logging.debug(f"Number of edges (label)\ntrain: {train_data['drug', 'protein'].edge_label.size()}")
    logging.debug(f"test: {test_data['drug', 'protein'].edge_label.size()}")
    logging.debug(f"val: {val_data['drug', 'protein'].edge_label.size()}")



    ## Run model
    logging.info(f'hidden channels: {hidden_channels}')

    model = Model(hidden_channels=hidden_channels, data=data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    early_stopper = EarlyStopper(tolerance=10, min_delta=0.05) 

    # lazy init, ned to rune one mode step for inferring number of parameters 
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    results_auc = [[],[],[]]
    results_aupr = [[],[],[]]
    loss_list = [[], [], []]

    init_time = time.time()

    for epoch in range(1_000): 
        loss = train(train_data)
        train_auc, _, _, train_loss, train_aupr = test(train_data)
        val_auc, _, _, val_loss, val_aupr = test(val_data)
        test_auc, emb, _, test_loss, test_aupr= test(test_data)
        if epoch%50 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_auc:.4f}, '
                    f'Val: {val_auc:.4f}, Test: {test_auc:.4f}')
        
        # list with AUC results
        results_auc[0].append(train_auc)
        results_auc[1].append(val_auc)
        results_auc[2].append(test_auc)

        # list with AUPR results
        results_aupr[0].append(train_aupr)
        results_aupr[1].append(val_aupr)
        results_aupr[2].append(test_aupr)

        # list with loss
        loss_list[0].append(train_loss)
        loss_list[1].append(val_loss)
        loss_list[2].append(test_loss)

        if early_stopper.early_stop(train_loss, val_loss) and epoch>40:    
            print('early stopped at epoch ', epoch)         
            break
    
    end_time = time.time()

    print(" ")
    print(f"Final AUC Train: {train_auc:.4f}, AUC Val {val_auc:.4f},AUC Test: {test_auc:.4f}")
    print(f"Final AUPR Train: {train_aupr:.4f}, AUC Val {val_aupr:.4f},AUC Test: {test_aupr:.4f}")
    print(f"Elapsed time {(end_time-init_time)/60:.4f} min")

    OUTPUT_PATH =  os.path.join('Results', f'{DATABASE.upper()}_{hidden_channels}' )

    # if not exists create it
    if not os.path.isdir(OUTPUT_PATH): 
        os.makedirs(OUTPUT_PATH)

    plot_auc(DATABASE, OUTPUT_PATH, results_auc, hidden_channels, 'auc')
    plot_auc(DATABASE, OUTPUT_PATH, results_aupr, hidden_channels, 'aupr')
    plot_auc(DATABASE, OUTPUT_PATH, loss_list, hidden_channels, 'loss')


    with open(os.path.join(OUTPUT_PATH, 'AUC_results.txt'), 'a') as f:
        f.write(f'{test_auc:.4f}\t{test_aupr:.4f}\t{(end_time-init_time)/60:.4f}\t{epoch}\n')


    np.save(f'{OUTPUT_PATH}/results_auc.npy', results_auc)
    np.save(f'{OUTPUT_PATH}/results_aupr.npy', results_aupr)
    np.save(f'{OUTPUT_PATH}/loss_list.npy', loss_list)
    np.save(f'{OUTPUT_PATH}/embeddings_drugs.npy', emb['drug'].cpu().numpy())
    np.save(f'{OUTPUT_PATH}/embeddings_proteins.npy', emb['protein'].cpu().numpy())

    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'model.pt'))

#####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
	main()
#####-------------------------------------------------------------------------------------------------------------