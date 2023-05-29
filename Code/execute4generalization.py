"""
Code for perform generalization evaluation.

First executes the training of the model, and saves in Results folder
returns pkl with results and heatmaps

function to Sh in func_test_models
"""

import argparse
import logging
import subprocess as sp

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os 

from function4testdata import sh_evaluation 
from utils import plot_heatmap


#################################
#################################

### CHANNELS
hidden_channels =  17

print(f'Embedding dimension {hidden_channels}')

FOLDER = 'Results/' # output folder


### REPETITION
datasets_o = ['DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']
datasets_t = datasets_o

nreps_training = 5
nreps = 5 


for rep_t in tqdm(range(nreps_training), desc='Repetitions'):

    for database1 in datasets_o:

        print('======================')
        print(f'Working with {database1}')

        for database2 in datasets_t:
            if database2 != database1:
                database = f'{database1}' + '_WO_' + f'{database2}'
                print(database)

                try:
                    print(f'database {database}, hidden_channels {hidden_channels}')
                    return_code = sp.check_call(['python3', 'Code/main.py', '-d', f'{database}', '-e', f'{hidden_channels}'])
                    if return_code ==0: 
                        print(f'EXIT CODE 0 FOR {database.upper()} with hd {hidden_channels}')

                except sp.CalledProcessError as e:
                    logging.info(e.output)
            
            else:
                database = f'{database1}'
                print(database)
                
                try:
                    print(f'database {database}, hidden_channels {hidden_channels}')
                    return_code = sp.check_call(['python3', 'Code/main.py', '-d', f'{database}', '-e', f'{hidden_channels}'])
                    if return_code ==0: 
                        print(f'EXIT CODE 0 FOR {database.upper()} with hd {hidden_channels}')

                except sp.CalledProcessError as e:
                    logging.info(e.output)
            


        print(f'Finished with {database1}')

    df_auc_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
    df_aupr_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)

    for rep in range(nreps):
        for dataset_trained in datasets_o:
            for dataset_evaluated in datasets_t:
                auc_list = [] 
                aupr_list = []
                auc, aupr = sh_evaluation(DATABASE_TRAINED= dataset_trained, DATABASE_EVAL= dataset_evaluated,  hd= hidden_channels)
                df_auc_hmp[dataset_trained].loc[dataset_evaluated] = auc
                df_aupr_hmp[dataset_trained].loc[dataset_evaluated] = aupr
    
        print(f"==== {hidden_channels} ====")

        print('AUC')
        print(df_auc_hmp)

        print('AUPR')
        print(df_aupr_hmp)

        # save data
        if not os.path.isdir(FOLDER): 
            os.makedirs(FOLDER)
        df_auc_hmp.to_pickle(f'{FOLDER}tmp_auc_{hidden_channels}_{rep_t}_{rep}.pkl')
        df_aupr_hmp.to_pickle(f'{FOLDER}tmp_aupr_{hidden_channels}_{rep_t}_{rep}.pkl')


print('finished!!!')


df_auc_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
df_aupr_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)

df_auc_std_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
df_aupr_std_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)

for dataset_trained in datasets_o:
    for dataset_evaluated in datasets_t:
        auc_list, aupr_list = [], []
        for rep_t in range(nreps_training):
            for rep in range(nreps):
                (rep_t, rep)
                # AUC
                df_auc = pd.read_pickle(f'{FOLDER}tmp_auc_{hidden_channels}_{rep_t}_{rep}.pkl')
                auc = df_auc[dataset_trained].loc[dataset_evaluated]
                auc_list.append(auc)
                # AUPR
                df_aupr = pd.read_pickle(f'{FOLDER}tmp_aupr_{hidden_channels}_{rep_t}_{rep}.pkl')
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
df_auc_hmp.to_pickle(f'{FOLDER}auc_{hidden_channels}.pkl')
df_aupr_hmp.to_pickle(f'{FOLDER}aupr_{hidden_channels}.pkl')
df_auc_std_hmp.to_pickle(f'{FOLDER}auc_std_{hidden_channels}.pkl')
df_aupr_std_hmp.to_pickle(f'{FOLDER}aupr_std_{hidden_channels}.pkl')

# df_auc_hmp = pd.read_pickle(f'{FOLDER}auc_{hidden_channels}.pkl')
# df_aupr_hmp = pd.read_pickle(f'{FOLDER}aupr_{hidden_channels}.pkl')
# df_auc_std_hmp = pd.read_pickle(f'{FOLDER}auc_std_{hidden_channels}.pkl')
# df_aupr_std_hmp = pd.read_pickle(f'{FOLDER}aupr_std_{hidden_channels}.pkl')

## PLOT HEATMAP
plot_heatmap(df_auc_hmp, df_auc_std_hmp, cmap='Oranges')
plt.savefig(f'{FOLDER}results_AUC_{hidden_channels}_.pdf',bbox_inches='tight',  pad_inches = 0.2)
plot_heatmap(df_aupr_hmp, df_aupr_std_hmp,'Oranges')
plt.savefig(f'{FOLDER}results_AUPR_{hidden_channels}_.pdf', bbox_inches='tight',  pad_inches = 0.2)
