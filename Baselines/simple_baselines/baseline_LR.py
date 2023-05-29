# execute using the same docker to have everything in the same torch version

import torch
import torch_geometric.transforms as T
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

database = 'drugbank'

print(f"Executing Linear regression For {database} & doing Grid Search")

# add line cuda if cuda available

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

train_data, _, test_data = split(data)



def get_X_y(set_data):

    drug_fts = set_data['drug'].x#.numpy()
    protein_fts = set_data['protein'].x#.numpy()

    edges = set_data['drug', 'protein'].edge_label_index

    edges_ft = torch.cat([drug_fts[edges[0]], protein_fts[edges[1]]], dim=-1)
    edge_label = set_data['drug', 'protein'].edge_label

    return edges_ft, edge_label


##

X_train, y_train = get_X_y(train_data)
X_test, y_test = get_X_y(test_data)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import roc_auc_score, average_precision_score



parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(0,3,7), # Inverse of regularization strength
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
    'max_iter' : [100, 500]
}


logreg = LogisticRegression()

gridsearch = GridSearchCV(logreg,                    # model
                    param_grid = parameters,   # hyperparameters
                    scoring = 'accuracy',        # metric for scoring
                    cv = 10,  # number of folds
                    verbose= 5,
                    n_jobs = -1,
                    return_train_score=True)                    


a = time.time()
gridsearch.fit(X_train,y_train)
b = time.time()


print('\nElapsed time gridsearch (min): ', (b-a)/60)


df_resul_cv = pd.DataFrame(gridsearch.cv_results_)
df_resul_cv.to_csv('Figures/logres/rndfrst_cv_results_.tsv', sep='\t')

# violin plots
search_values = ['param_'+value for value in list(parameters.keys())]

for search_value in search_values:
    search_value
    plt.clf()
    sns.violinplot(data=df_resul_cv, x=search_value, y='mean_train_score', palette=sns.color_palette("pastel"))
    sns.violinplot(data=df_resul_cv, x=search_value, y='mean_test_score')
    plt.savefig(f'Figures/logres/logres_{search_value}.pdf', dpi=330)





print('=== best estimator ===')
print(gridsearch.best_estimator_)
print("Tuned Hyperparameters (best_params_):")
print(gridsearch.best_params_)
print("Accuracy :\n", gridsearch.best_score_)


# set best model
final_logres = gridsearch.best_estimator_


print("Running the selected model 5 times with different set of edges")

aucs, auprs, times = [], [], []


for it in range(5):
    init = time.time()
    train_data, _, test_data = split(data) # don't take val now

    X_train, y_train = get_X_y(train_data)
    X_test, y_test = get_X_y(test_data)

    model = final_logres.fit(X_train, y_train)

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

    model = final_logres.fit(X_train, y_train) # train the model in new dataset with (same) best parameters 

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

