import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# load 

PATH_OUT = 'Results/5th_validation/'

datasets = ['DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']

df_percent = pd.DataFrame(columns=datasets, index=datasets)
df = pd.DataFrame(columns=datasets, index=datasets)

df_percent_zeros = pd.DataFrame(columns=datasets, index=datasets)
df_zeros = pd.DataFrame(columns=datasets, index=datasets)


# df_nodes_drugs = pd.DataFrame(columns=datasets, index=datasets)
# df_nodes_drugs_perc = pd.DataFrame(columns=datasets, index=datasets)
# df_nodes_proteins = pd.DataFrame(columns=datasets, index=datasets)
# df_nodes_proteins_perc = pd.DataFrame(columns=datasets, index=datasets)


for database1 in datasets:
    for database2 in datasets:
        print(f"----- {database1};{database2} -----")
        # Load Datasets
        dataset1 = pd.read_pickle(os.path.join('Data', database1.upper(), f'dti_{database1.lower()}.pkl')).astype(str).drop_duplicates(keep='first')
        dataset2 = pd.read_pickle(os.path.join('Data', database2.upper(), f'dti_{database2.lower()}.pkl')).astype(str).drop_duplicates(keep='first')

        ########## NODES (this is new...)
        shared_drug_nodes = len(set(dataset1.Drug).intersection(set(dataset2.Drug)))
        print(f'Dataset 1 {database1} has {shared_drug_nodes} drug_nodes edges that also appear in {database2}')
        shared_protein_nodes = len(set(dataset1.Protein).intersection(set(dataset2.Protein)))
        print(f'Dataset 1 {database1} has {shared_protein_nodes} protein_nodes edges that also appear in {database2}')
        # df_nodes_drugs[database1].loc[database2] = shared_drug_nodes
        # df_nodes_proteins[database1].loc[database2] = shared_protein_nodes

        ########### EDGES
        ## POSITIVES
        # DATASET 1 is DATASET TRAINED (columns), DATASET 2 is DATASET TEST (rows)
        # We need to extrac the common (positive) edges
        dataset_1_dtis = set((dataset1['Drug'] + '-' + dataset1["Protein"]))
        dataset2_positive_edges = set((dataset2['Drug'] + '-' + dataset2["Protein"]))

        #calculate the intersection
        repeated_positives = dataset_1_dtis.intersection(dataset2_positive_edges)
        num_repeated_positives = len(repeated_positives)
        print(f'Dataset 1 {database1} has {num_repeated_positives} positive edges that also appear in {database2}')
        print(f'A {num_repeated_positives/dataset1.shape[0]*100:.2f}% of Dataset 1 is present in Dataset 2')

        ## NEGATIVES
        # DATASET 1 is DATASET TRAINED (columns), DATASET 2 is DATASET TEST (rows)
        # We need to extract from Dataset Trained those 0s that are 1s in Dataset 2
        # in this case only counting

        # then lets generate zeros for dataset 1 
        dataset1_all_edges = []
        drugs_dataset1 = set(dataset1.Drug)
        proteins_dataset1 = set(dataset1.Protein)

        for drug in tqdm(drugs_dataset1, desc='Retrieving all posible edges'):
            #print(drug)
            for protein in proteins_dataset1:
                #(drug,protein)
                dataset1_all_edges.append(f'{drug}-{protein}')

        dataset1_all_edges = set(dataset1_all_edges)
        dataset1_negative_edges = dataset1_all_edges- dataset_1_dtis
        assert len(dataset1_all_edges) == len(dataset_1_dtis) + len(dataset1_negative_edges) , 'missing edges'

        # compare those 0s in db1 that are ones in db2
        false_negatives_dataset1 = dataset1_negative_edges.intersection(dataset2_positive_edges)
        print('10 example')
        print(list(false_negatives_dataset1)[:10])
        negatives_in_1_that_reported_positive_in_2 = len(false_negatives_dataset1)

        # compare which of those 0s are 1s in dataset 2
        print(f'{database1} has {negatives_in_1_that_reported_positive_in_2} negative edges that were reported as positive in {database2}')
        print(f'This are a {negatives_in_1_that_reported_positive_in_2/len(dataset1_negative_edges)*100:.2f}% of {database1} zeros')

        if false_negatives_dataset1 != set():
            PATH_VAL = PATH_OUT + f'df_validation_for_{database1.lower()}_originally_{database2.lower()}.tsv'
            print(f'Saving this list in {PATH_VAL}')

            df_validation = pd.DataFrame(false_negatives_dataset1, columns=['tmp'])
            df_validation = df_validation.tmp.str.split('-', expand=True)
            df_validation.columns = ['Drug', 'Protein']
            df_validation.to_csv(PATH_VAL, sep='\t')

        value_ones = num_repeated_positives 
        value_zeros = negatives_in_1_that_reported_positive_in_2

        # load in new dataframe for ones
        df_percent[database1].loc[database2] = value_ones/dataset1.shape[0]
        df[database1].loc[database2] = value_ones

        # load in new dataframe for ones
        df_percent_zeros[database1].loc[database2] = value_zeros/dataset1.shape[0]
        df_zeros[database1].loc[database2] = value_zeros
        
        print(' ')



print(df_zeros)

df.to_pickle(PATH_OUT + 'df_repeated.pkl')
df_percent.to_pickle(PATH_OUT + 'df_repeated_percent.pkl')
df_zeros.to_pickle(PATH_OUT + 'zeros.pkl')
df_percent_zeros.to_pickle(PATH_OUT + 'df_percent_zeros.pkl')


# df_nodes_drugs.to_pickle(PATH_OUT + 'shared_drugs.pkl')
# df_nodes_proteins.to_pickle(PATH_OUT + 'shared_proteins.pkl')

############################################
# Join datasets for each dataset

import glob


for database1 in datasets:

    path_val = PATH_OUT + f'df_validation_for_{database1.lower()}_originally_*.tsv'
    list_join = glob.glob(path_val)
    df_join = pd.DataFrame(columns=['Drug', 'Protein'])

    for path_join in list_join:
        #path_join
        dfj = pd.read_csv(path_join, sep='\t').drop(columns='Unnamed: 0')
        #dfj.shape
        df_join = pd.concat([df_join, dfj])

    df_join = df_join.drop_duplicates().dropna()

    df_join.shape
    path_out = PATH_OUT + f'validation_joint_{database1.lower()}.tsv'

    df_join.to_csv(path_out, sep="\t",index=None)






#############################################

## PLOTS
# print('ended')
# plt.clf()
# fig = plt.figure()
# fig.set_size_inches(10, 10)
# df = df.astype(float)
# ax = sns.heatmap(df, annot=True, vmin=0, vmax= 50, cmap='Reds', fmt=".0f")
# ax.set(xlabel="Trained", ylabel="Evaluated")
# ax.xaxis.tick_top()
# plt.savefig('../repeated_edges.pdf')
# df_percent = pd.read_pickle(PATH_OUT + 'df_repeated_percent.pkl')

# plt.clf()
# fig = plt.figure()
# fig.set_size_inches(11, 10)
# df_percent = df_percent.astype(float)
# ax = sns.heatmap(df_percent, annot=True, vmin=0, vmax= 1, cmap='RdPu', fmt=".1%", annot_kws={"size":11, "fontweight":'semibold'})
# #ax.set(xlabel="Trained", ylabel="Evaluated")
# ax.xaxis.tick_top()
# ax.tick_params(labelsize=12)
# plt.savefig(PATH_OUT + 'heatmap_repeated_edges.pdf',bbox_inches='tight',  pad_inches = 0.2)


# df_zeros = pd.read_pickle(PATH_OUT + 'zeros.pkl')
# df_zeros

# plt.clf()
# fig = plt.figure()
# fig.set_size_inches(10, 10)
# df_zeros = df_zeros.astype(float)
# ax = sns.heatmap(df_zeros, annot=True, vmin=0, vmax= 1, cmap='RdPu', fmt=".0f", cbar=False, annot_kws={"size":13, "fontweight":'semibold'})
# #ax.set(xlabel="Trained", ylabel="Evaluated")
# ax.xaxis.tick_top()
# ax.tick_params(labelsize=12)
# plt.savefig(PATH_OUT + 'heatmap_zeros.pdf', bbox_inches='tight',  pad_inches = 0.2)


# plt.clf()
# fig = plt.figure()
# fig.set_size_inches(10, 10)
# df_nodes_drugs = df_nodes_drugs.astype(float)
# ax = sns.heatmap(df_nodes_drugs, annot=True, vmin=0, vmax= 1, cmap='RdPu', fmt=".0f", cbar=False, annot_kws={"size":13, "fontweight":'semibold'})
# #ax.set(xlabel="Trained", ylabel="Evaluated")
# ax.xaxis.tick_top()
# ax.tick_params(labelsize=12)
# plt.savefig(PATH_OUT + 'heatmap_shared_drugs.pdf', bbox_inches='tight',  pad_inches = 0.2)


# plt.clf()
# fig = plt.figure()
# fig.set_size_inches(10, 10)
# df_nodes_proteins = df_nodes_proteins.astype(float)
# ax = sns.heatmap(df_nodes_proteins, annot=True, vmin=0, vmax= 1, cmap='RdPu', fmt=".0f", cbar=False, annot_kws={"size":13, "fontweight":'semibold'})
# #ax.set(xlabel="Trained", ylabel="Evaluated")
# ax.xaxis.tick_top()
# ax.tick_params(labelsize=12)
# plt.savefig(PATH_OUT + 'heatmap_shared_proteins.pdf', bbox_inches='tight',  pad_inches = 0.2)
