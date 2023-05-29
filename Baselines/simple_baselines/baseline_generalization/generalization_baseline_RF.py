import numpy as np
import torch
import torch_geometric.transforms as T

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def get_X_y(set_data):
    
    drug_fts = set_data['drug'].x
    protein_fts = set_data['protein'].x

    edges = set_data['drug', 'protein'].edge_label_index

    edges_ft = torch.cat([drug_fts[edges[0]], protein_fts[edges[1]]], dim=-1)
    edge_label = set_data['drug', 'protein'].edge_label

    return edges_ft, edge_label




def train_model(path_train_data_gennius):

    split = T.RandomLinkSplit(
        num_val= 0.0, # remove here next
        num_test= 0.2, 
        is_undirected= True,
        add_negative_train_samples= True, # False for: Not adding negative links to train
        neg_sampling_ratio= 1.0, # ratio of negative sampling is 0
        #disjoint_train_ratio = 0.2, # not needed now
        edge_types=[('drug', 'interaction', 'protein')],
        rev_edge_types=[('protein', 'rev_interaction', 'drug')],
        split_labels=False
        )

    data = torch.load(path_train_data_gennius)
    data.to('cpu')

    train_data, _, test_data = split(data) # don't take val now

    X_train, y_train = get_X_y(train_data)
    X_test, y_test = get_X_y(test_data)


    model = RF.fit(X_train, y_train)

    prediction = model.predict(X_test)

    auc = roc_auc_score(y_test, prediction)
    aupr = average_precision_score(y_test, prediction)
    print(f'Trained model returned: AUC {auc}, AUPR: {aupr}')

    return model



def test_dataset(model, path_test_data_gennius):

    split_all_test = T.RandomLinkSplit(
                num_val= 0.0, # remove here next
                num_test= 0.0, 
                is_undirected= True,
                add_negative_train_samples= True, # False for: Not adding negative links to train
                neg_sampling_ratio= 1.0, # ratio of negative sampling is 0
                #disjoint_train_ratio = 0.2, # not needed now
                edge_types=[('drug', 'interaction', 'protein')],
                rev_edge_types=[('protein', 'rev_interaction', 'drug')],
                split_labels=False
                )

    data_pyg = torch.load(path_test_data_gennius)
    data_pyg.to('cpu')

    data, _, _ = split_all_test(data_pyg) # don't take val now

    X_test_dataset, y_test_dataset = get_X_y(data)

    prediction_gen = model.predict(X_test_dataset)

    auc = roc_auc_score(y_test_dataset, prediction_gen)
    aupr = average_precision_score(y_test_dataset, prediction_gen)

    return auc, aupr




# selected with gridsearch
RF = RandomForestClassifier(criterion='gini', max_depth=10, max_features = 'auto',
                        min_samples_leaf=6, min_samples_split=5, n_estimators=50)



# DATABASE_TRAINED = 'drugbank'
# DATABASE_EVAL = 'biosnap'



datasets_o = ['DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']
datasets_t = datasets_o

# nreps_training = 5
# nreps = 5 

# print(f'For training use: {FULL_NAME_TRAINED}')
# print(f'For eval we need {DATABASE_EVAL}')


#for rep_t in tqdm(range(nreps_training), desc='Repetitions'):
# TRAINING LOOP

for rep_train in range(5):

    df_auc_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
    df_aupr_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)

    for database_train in datasets_o:

        print('======================')
        print(f'Working with {database_train}')

        for database_test in datasets_t:

            # different
            if database_train != database_test:
                database = f'{database_train}' + '_WO_' + f'{database_test}'

            # same will overfit later remove for mean values
            else:
                database = f'{database_train}'
            
            # here we train and test one time
            path_train_data_gennius = f'Data/{database_train.upper()}/hetero_data_{database_train.lower()}.pt'
            model = train_model(path_train_data_gennius)

            ### TESTING
            # but now we need to load the test data to report that value

            path_test_data_gennius = f'Data/{database_test.upper()}/hetero_data_{database_test.lower()}.pt'
            
            # create loop, data only needs to be loaded once:
            auc, aupr = test_dataset(model, path_test_data_gennius)

            df_auc_hmp[database_train].loc[database_test] = auc
            df_aupr_hmp[database_train].loc[database_test] = aupr

            print('TEST:')
            print(database)
            print(auc, aupr)

    print('finished!!!')
    print(df_auc_hmp)
    print(df_aupr_hmp)
    df_auc_hmp.to_pickle(f'Baselines/baseline_generalization/df_auc_hmp_{rep_train}.pkl')
    df_aupr_hmp.to_pickle(f'Baselines/baseline_generalization/df_aupr_hmp_{rep_train}.pkl')



# path_train_data_gennius = f'../../GENNIUS/Data/{FULL_NAME_TRAINED.upper()}/hetero_data_{FULL_NAME_TRAINED.lower()}.pt'

# model = train_model(path_train_data_gennius)

# ### TESTING
# # but now we need to load the test data to report that value

# path_test_data_gennius = f'../../GENNIUS/Data/{DATABASE_EVAL.upper()}/hetero_data_{DATABASE_EVAL.lower()}.pt'


# auc, aupr = test_dataset(model, path_test_data_gennius)

# print('TEST:')
# print(auc, aupr)


## JOINING ALL

df_auc_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
df_aupr_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)

df_auc_std_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
df_aupr_std_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)

for dataset_trained in datasets_o:
    for dataset_evaluated in datasets_t:
        auc_list, aupr_list = [], []
        for rep_train in range(5):
                # AUC
                #df_auc = pd.read_pickle(f'{FOLDER}tmp_auc_{hidden_channels}_{rep_t}_{rep}.pkl')
                df_auc = pd.read_pickle(f'Baselines/baseline_generalization/df_auc_hmp_{rep_train}.pkl')
                auc = df_auc[dataset_trained].loc[dataset_evaluated]
                auc_list.append(auc)
                # AUPR
                #df_aupr = pd.read_pickle(f'{FOLDER}tmp_aupr_{hidden_channels}_{rep_t}_{rep}.pkl')
                df_aupr = pd.read_pickle(f'Baselines/baseline_generalization/df_aupr_hmp_{rep_train}.pkl')
                aupr = df_aupr[dataset_trained].loc[dataset_evaluated]
                aupr_list.append(aupr)
        # end loop repetition
        final_auc = np.array(auc_list).mean()
        final_auc_std = np.array(auc_list).std()
        final_aupr = np.array(aupr_list).mean()
        final_aupr_std = np.array(aupr_list).std() 
        # ADD HEATMAP
        df_auc_hmp[dataset_trained].loc[dataset_evaluated] = final_auc
        df_aupr_hmp[dataset_trained].loc[dataset_evaluated] = final_aupr
        df_auc_std_hmp[dataset_trained].loc[dataset_evaluated] = final_auc_std
        df_aupr_std_hmp[dataset_trained].loc[dataset_evaluated] = final_aupr_std



# save data
# save data
df_auc_hmp.to_pickle(f'Baselines/baseline_generalization/auc_mean.pkl')
df_aupr_hmp.to_pickle(f'Baselines/baseline_generalization/aupr_mean.pkl')
df_auc_std_hmp.to_pickle(f'Baselines/baseline_generalization/auc_std.pkl')
df_aupr_std_hmp.to_pickle(f'Baselines/baseline_generalization/aupr_std.pkl')
