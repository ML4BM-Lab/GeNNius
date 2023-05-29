import os
import numpy as np
import pandas as pd
import json

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score

from GeNNius import Model, EarlyStopper, shuffle_label_data


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


def get_negatives(set_selection, edges_avoid, sampling_ratio):

    edges = set_selection['drug', 'protein'].edge_label_index.cpu().numpy()
    labels_ = set_selection['drug', 'protein'].edge_label.cpu().numpy()

    index_label_0_ = np.where(labels_ == 0)[0]

    edges_to_change  = edges[:,index_label_0_]

    a = edges_avoid
    b = edges_to_change

    # Create a boolean mask indicating which pairs in b are not already in a
    mask = np.ones((b.shape[1],), dtype=bool)
    for i in range(b.shape[1]):
        pair = b[:, i]
        if np.any(np.all(a == pair.reshape((2, 1)), axis=0)):
            mask[i] = False

    # Create a shortened version of b containing only the unique pairs that are not in a
    b_shortened = np.unique(b[:, mask], axis=1)
    b_shortened = b_shortened[:, :int(edges_to_change.shape[1]/sampling_ratio)]  # Truncate b_shortened to the same size as a

    print('re-check worked!')
    for i in range(b_shortened.shape[1]):
        pair = b_shortened[:, i]
        if np.any(np.all(a == pair.reshape((2, 1)), axis=0)):
            print('appears')
            ValueError
    
    return b_shortened


def change_dataobject(splitted_dataobject, new_negs):

    edges_train = splitted_dataobject['drug', 'protein'].edge_label_index.cpu().numpy()
    labels_ = splitted_dataobject['drug', 'protein'].edge_label.cpu().numpy()

    index_label_1_ = np.where(labels_ == 1)[0] # labels 1
    index_label_0_ = np.where(labels_ == 0)[0] # labels 0
    index_label_0_cut = index_label_0_[:index_label_1_.shape[0]]

    #concatenate
    edges_train_label_1_  = edges_train[:,index_label_1_]
    edges_train_label_0_ = new_negs

    final_train_edges = np.concatenate((edges_train_label_1_, edges_train_label_0_), axis=1)
    final_train_labels = np.concatenate((labels_[index_label_1_], labels_[index_label_0_cut]), axis=0)

    assert final_train_edges.shape[1] == final_train_labels.shape[0], 'shapes failing'


    print('re-check worked!')

    for i in range(edges_train_label_0_.shape[1]):
        pair = edges_train_label_0_[:, i]
        if np.any(np.all(edges_train_label_1_ == pair.reshape((2, 1)), axis=0)):
            print('appears')
            ValueError
    
    # Make sure type is #'torch.cuda.FloatTensor'
    final_train_edges = torch.from_numpy(final_train_edges).cuda()
    final_train_labels = torch.from_numpy(final_train_labels).cuda()

    splitted_dataobject['drug', 'protein'].edge_label_index = final_train_edges
    splitted_dataobject['drug', 'protein'].edge_label = final_train_labels

    return splitted_dataobject


####################################################
####################################################

dataset = 'biosnap'.lower() # indicate here the dataset

print(dataset)

not_see_val = True


hd = 17 # hidden dimensions

device = 'cuda'

PATH_VAL = 'Results/5th_validation/' # path for validation files

####################################################
####################################################

# GET THE DATASET FOR VALIDATION
path_dti_val = os.path.join(PATH_VAL, f'validation_joint_{dataset.lower()}.tsv')

dti_val = pd.read_csv(path_dti_val, sep='\t')
dti_val['Drug'] = dti_val.Drug.astype(str) # safety
dti_val.shape


# We do not need to generate the whole datset as we have already dictionaries in DATASET/
# from when we loaded the whole dataset !!
PATH_DATA = os.path.join('Data', dataset.upper(), f'hetero_data_{dataset}.pt')

data_model = torch.load(PATH_DATA)

with open(os.path.join('Data', dataset.upper(), 'drug_mapping.json')) as f:
    drug_mapping = json.load(f)


with open(os.path.join('Data', dataset.upper(),'protein_mapping.json')) as f:
    protein_mapping = json.load(f)



# Generate Object for validation
data_validate = HeteroData()

data_validate['drug'].x = data_model['drug'].x
data_validate['protein'].x = data_model['protein'].x 

src = [drug_mapping[index] for index in dti_val['Drug']]
dst = [protein_mapping[index] for index in dti_val['Protein']]
edge_index_validation = torch.tensor([src, dst])
data_validate['drug', 'interaction', 'protein'].edge_index = edge_index_validation

print(data_validate)
data_validate.to(device)


### Now we need to train the model again
# but making sure that we do not see as negatives cercain values when training or validating

# Prepare data for splitting
data_model = T.ToUndirected()(data_model)
# Remove "reverse" label.
del data_model['protein', 'rev_interaction', 'drug'].edge_label  


sampling_ratio = 3.0

split = T.RandomLinkSplit(
    num_val= 0.1,
    num_test= 0.0, # we dont need test now, we can use the whole network 
    is_undirected= True,
    add_negative_train_samples= True, # False for: Not adding negative links to train
    neg_sampling_ratio= sampling_ratio, # ratio of negative sampling is 0
    disjoint_train_ratio = 0.2, #
    edge_types=[('drug', 'interaction', 'protein')],
    rev_edge_types=[('protein', 'rev_interaction', 'drug')],
    split_labels=False
)


if not_see_val:
    print('Avoiding edges to validate during train and val')
    train_data_, val_data_, _ = split(data_model)

    # modifies train
    new_negs_train =  get_negatives(train_data_, edge_index_validation, sampling_ratio) 
    train_data = change_dataobject(train_data_, new_negs_train)
    #train_data.to(device) # to ensure it didnt changed sstuff

    # modifies val
    new_negs_val = get_negatives(val_data_, edge_index_validation, sampling_ratio)
    val_data = change_dataobject(val_data_, new_negs_val)

else:
    print('Completely random')
    train_data, val_data, _ = split(data_model)


### NOW NEED TO TRAIN THE MODEL

model = Model(hidden_channels=hd, data=data_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

early_stopper = EarlyStopper(tolerance=10, min_delta=0.05)

for epoch in range(1_000): 
    loss = train(train_data)
    train_auc, _, _, train_loss, train_aupr = test(train_data)
    val_auc, _, _, val_loss, val_aupr = test(val_data)

    if epoch%100 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_auc:.4f}, 'f'Val: {val_auc:.4f}')
    
    if early_stopper.early_stop(train_loss, val_loss) and epoch>40:    
        print('early stopped at epoch ', epoch)         
        break

print(f'Train AUC: {train_auc:.4f}, Val AUC {val_auc:.4f}')
print(f'Train AUPR: {train_aupr:.4f}, Val AUPR {val_auc:.4f}')


## NOW MODEL IS TRAINED
# NEED TO TEST IN SPECIFIC DATA

data_validate
data_validate = T.ToUndirected()(data_validate)
del data_validate['protein', 'rev_interaction', 'drug'].edge_label  # Remove "reverse" label.

data_validate['drug', 'protein'].edge_label_index = data_validate['drug', 'protein'].edge_index



with torch.no_grad():
    model.eval()
    _, pred = model(data_validate.x_dict, data_validate.edge_index_dict,
                data_validate['drug', 'protein'].edge_label_index)
    out = pred.view(-1).sigmoid()



# should work adding it directly from pandas df but to ensure that no odering was mod
# 1 or 2 lines more its ok

out_drug_num = data_validate['drug', 'protein'].edge_index.cpu().numpy()[0]
out_drug_prot = data_validate['drug', 'protein'].edge_index.cpu().numpy()[1]

rev_drug_mapping = {value: key for key, value in drug_mapping.items()}
rev_protein_mapping = {value: key for key, value in protein_mapping.items()}

out_drugs = [rev_drug_mapping.get(drug) for drug in out_drug_num]
out_prots = [rev_protein_mapping.get(prot) for prot in out_drug_prot]
out_preds = out.cpu().numpy()

results = pd.DataFrame({'Drug': out_drugs, 'Protein': out_prots, 'Predictions': out_preds})

filename = f'results_{dataset}.tsv'

# Check if the file already exists
if os.path.isfile(filename):

    i = 1
    while True:
        new_filename = f"{os.path.splitext(filename)[0]}_{i}.tsv"
        if os.path.isfile(new_filename):
            i += 1
        else:
            filename = new_filename
            break

results.to_csv(os.path.join(PATH_VAL, filename), sep="\t", index=None)


predicted_rate_05 = round(sum([1 if value>=0.5 else 0 for value in out])/len(out), 4)
print(f'WITH THRESHOLD 0.5 predicted {predicted_rate_05}')

