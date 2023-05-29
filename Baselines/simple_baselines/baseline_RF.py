# execute using the same docker to have everything in the same torch version
# and pyg version

# bslins
# baseline random forest
# docker run -dt --gpus all -v /home/uveleiro/data/uveleiro/ALL_GENNIUS:/wdir/ --name bslins genniusdoc

import numpy as np
import torch
import torch_geometric.transforms as T

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


database = 'drugbank'


print("Executing Random Forest For DrugBank & doing Grid Search")

# add line cuda if cuda available
# genniusdoc

path_data_gennius = f'../GENNIUS/Data/{database.upper()}/hetero_data_{database.lower()}.pt'

data = torch.load(path_data_gennius)
data.to('cpu')

# using the same splitting

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

train_data, _, test_data = split(data) # don't take val now


def get_X_y(set_data):

    drug_fts = set_data['drug'].x
    protein_fts = set_data['protein'].x

    edges = set_data['drug', 'protein'].edge_label_index

    edges_ft = torch.cat([drug_fts[edges[0]], protein_fts[edges[1]]], dim=-1)
    edge_label = set_data['drug', 'protein'].edge_label

    return edges_ft, edge_label


##

X_train, y_train = get_X_y(train_data)
X_test, y_test = get_X_y(test_data)

#########


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV
import time

# model = RandomForestClassifier(random_state= 101).fit(X_train,y_train)
# predictionforest = model.predict(X_test)
# print(confusion_matrix(y_test,predictionforest))
# print(classification_report(y_test,predictionforest))
# acc1 = accuracy_score(y_test, predictionforest)
# auc = roc_auc_score(y_test, predictionforest)
# aupr = average_precision_score(y_test, predictionforest)
# print(auc, aupr)


grid_search = {'criterion': ['entropy', 'gini'],
               'max_depth': [2, 6, 8, 10],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [4, 6, 8, 10, 20],
               'min_samples_split': [2, 5, 7, 10, 20],
               'n_estimators': [10, 20, 30, 50]}

clf = RandomForestClassifier()

gridsearch = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 10, verbose= 5, n_jobs = -1, return_train_score=True)

a = time.time()
gridsearch.fit(X_train,y_train)
b = time.time()

print('Elapsed time gridsearch (min): ', (b-a)/60)


# table = pd.pivot_table(pd.DataFrame(model.cv_results_),
#     values='mean_test_score', index='param_n_estimators', 
#                        columns='param_max_depth')


df_resul_cv = pd.DataFrame(gridsearch.cv_results_)
df_resul_cv.to_csv('Figures/rndfrst/rndfrst_cv_results_.tsv', sep='\t')

# violin plots

search_values = ['param_'+value for value in list(grid_search.keys())]

for search_value in search_values:
    search_value
    plt.clf()
    #sns.heatmap(table, cmap="crest")
    sns.violinplot(data=df_resul_cv, x=search_value, y='mean_train_score', palette=sns.color_palette("pastel"))
    sns.violinplot(data=df_resul_cv, x=search_value, y='mean_test_score')
    plt.savefig(f'Figures/rndfrst/test_rndfs_{search_value}.pdf', dpi=330)


print('=== best estimator ===')
print(gridsearch.best_estimator_)
print("Tuned Hyperparameters (best_params_):")
print(gridsearch.best_params_)
print("Accuracy :\n", gridsearch.best_score_)


# dic_bp = gridsearch.best_params_
# final_rndfst = RandomForestClassifier(criterion = dic_bp['criterion'],
#                        max_depth = dic_bp['max_depth'],
#                        max_features = dic_bp['max_features'],
#                        min_samples_leaf = dic_bp['min_samples_leaf'],
#                        min_samples_split = dic_bp['min_samples_split'],
#                        n_estimators = dic_bp['n_estimators']
#                     )

# is the same?

final_rndfst = gridsearch.best_estimator_

# predictionforest = model.best_estimator_.predict(X_test)
# print(confusion_matrix(y_test,predictionforest))
# print(classification_report(y_test,predictionforest))


# once the best model is stimated, perform different splits to get an average of 5 independent runs

print("Running the selected model 5 times with different set of edges")

aucs, auprs, times = [], [], []

# missing testing again with best parameters ! !!

for it in range(5):
    init = time.time()
    train_data, _, test_data = split(data) # don't take val now

    X_train, y_train = get_X_y(train_data)
    X_test, y_test = get_X_y(test_data)

    model = final_rndfst.fit(X_train, y_train)

    predictionforest = model.predict(X_test)

    end = time.time()

    # get results
    auc = roc_auc_score(y_test, predictionforest)
    aupr = average_precision_score(y_test, predictionforest)
    el_time = (end-init)

    aucs.append(auc)
    auprs.append(aupr)
    times.append(el_time)

    print(f'it {it}. AUC: {auc:.4f}, AUPR: {aupr:.4f}, time: {el_time} s')

print(f"Final Results {database}:")
print('AUC', np.mean(aucs), np.std(aucs))
print('AUPR', np.mean(auprs), np.std(auprs))
print('Time (s)', np.mean(el_time), np.std(el_time))


database = 'BIOSNAP'
print(f"With the same model, results for {database}")

# Load again
path_data_gennius = f'../GENNIUS/Data/{database.upper()}/hetero_data_{database.lower()}.pt'
data = torch.load(path_data_gennius)
data.to('cpu')


aucs, auprs, times = [], [], []

for it in range(5):
    init = time.time()
    train_data, _, test_data = split(data) # splitting new data

    X_train, y_train = get_X_y(train_data) # returning edge sets in ok format
    X_test, y_test = get_X_y(test_data)

    model = final_rndfst.fit(X_train, y_train) # train the model in new dataset with (same) best parameters 

    predictionforest = model.predict(X_test)

    end = time.time()

    # get results
    auc = roc_auc_score(y_test, predictionforest)
    aupr = average_precision_score(y_test, predictionforest)
    el_time = (end-init)

    aucs.append(auc)
    auprs.append(aupr)
    times.append(el_time)

    print(f'it {it}. AUC: {auc:.4f}, AUPR: {aupr:.4f}, time: {el_time} s')

print(f"Final Results {database}:")
print('AUC', np.mean(aucs), np.std(aucs))
print('AUPR', np.mean(auprs), np.std(auprs))
print('Time (s)', np.mean(el_time), np.std(el_time))

