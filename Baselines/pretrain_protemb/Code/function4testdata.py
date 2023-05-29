import os
import logging
import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score
from GeNNius import Model


@torch.no_grad()
def test(model, data):
    model.eval()
    emb, pred = model(data.x_dict, data.edge_index_dict,
                 data['drug', 'protein'].edge_label_index)

    # target value
    target = data['drug', 'protein'].edge_label.float()
    
    out = pred.view(-1).sigmoid()
    auc = roc_auc_score(target.cpu().numpy(), out.detach().cpu().numpy())
    aupr = average_precision_score(target.cpu().numpy(), out.detach().cpu().numpy())
    
    return round(auc, 6), round(aupr,6), out, target



def sh_evaluation(DATABASE_TRAINED, DATABASE_EVAL, hd= 8):
    
    ### Load model
    logging.debug('------> Loading Model...')
    device = 'cuda'

    if DATABASE_TRAINED != DATABASE_EVAL:
        FULL_NAME_TRAINED = f'{DATABASE_TRAINED}_WO_{DATABASE_EVAL}'.lower()
    else:
        FULL_NAME_TRAINED = DATABASE_TRAINED

    PATH_MODEL = os.path.join(f'Results', f'{FULL_NAME_TRAINED.upper()}_{hd}', 'model.pt') 

    print(f'Using trained: {FULL_NAME_TRAINED}')

    PATH_DATA_EVAL = f'Data/{DATABASE_EVAL.upper()}/hetero_data_{DATABASE_EVAL.lower()}.pt'

    # Load data needed for model innit 

    logging.debug('----> Loading Data (trained)...')

    # Load Evaluation data
    data_eval = torch.load(PATH_DATA_EVAL)

    # Prepare data
    data_eval = T.ToUndirected()(data_eval)
    del data_eval['protein', 'rev_interaction', 'drug'].edge_label  # Remove "reverse" label.

    logging.debug('----> Loading Model (trained)...')

    # Load Model
    model = Model(hidden_channels=hd, data=data_eval).to(device)

    model.load_state_dict(torch.load(PATH_MODEL))
    model.eval()

    # Process data for evaluation, implies adding random negatives
    split_val = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples= True, 
        neg_sampling_ratio= 1.0, 
        edge_types=[('drug', 'interaction', 'protein')],
        rev_edge_types=[('protein', 'rev_interaction', 'drug')],
        split_labels=False
    )

    logging.debug('---> Splitting Eval')
    sp_data, _, _ = split_val(data_eval)

    logging.debug('---> Testing Eval')
    auc, aupr, _, _ =  test(model, sp_data)

    print(f'Trained in {DATABASE_TRAINED}, evaluating in {DATABASE_EVAL}, AUC: {auc}, AUPR: {aupr}')

    return auc, aupr



###########################################################



"""
# Example

DATABASE_TRAINED = 'DrugBank'
DATABASE_EVAL = 'E'
hd = 10

logging.info(f'working with {hd} hidden channels ')

sh_evaluation(DATABASE_TRAINED= 'drugbank', DATABASE_EVAL= 'nr',  hd= 10)

"""