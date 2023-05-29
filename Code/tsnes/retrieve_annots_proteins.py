import os
import pandas as pd
from tqdm import tqdm
import logging
import json

from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy.sql import text


def get_dict_chembl2uni(PATHDICT):
    logging.info('Retrieving dictionary chembl2uni from rest api')
    # original: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_30/chembl_uniprot_mapping.txt
    with open(PATHDICT) as f:
        a = f.readlines()
    a = [line.strip('\n') for line in a] 
    data = [tuple(cont.split('\t')[:2]) for cont in a[1:]] 
    chembl2uni = {chembl:uni for uni,chembl in data}
    return chembl2uni

##


PATH_CHEMBL = 'Data/Raw/ChEMBL_31/chembl_31.db'
PATHDICT='Data/Raw/ChEMBL_31/chembl_uniprot_mapping.txt'


qtext = """
SELECT
    target_dictionary.chembl_id          AS target_chembl_id,
    protein_family_classification.l1     AS l1,
    protein_family_classification.l2     AS l2,
    protein_family_classification.l3     AS l3,
    protein_family_classification.l4     AS l4
FROM target_dictionary
    JOIN target_components ON target_dictionary.tid = target_components.tid
    JOIN component_class ON target_components.component_id = component_class.component_id
    JOIN protein_family_classification ON component_class.protein_class_id = protein_family_classification.protein_class_id
"""


#JOIN PROTEIN_CLASSIFICATION
# load
engine = create_engine(f'sqlite:///{PATH_CHEMBL}')

with engine.begin() as conn:
    res = conn.execute(text(qtext))
    df_chembl = pd.DataFrame(res.fetchall())

df_chembl.columns = res.keys()
df_chembl = df_chembl.where((pd.notnull(df_chembl)), None)


chembl2uni = get_dict_chembl2uni(PATHDICT)
df_chembl['UniprotID'] = df_chembl.target_chembl_id.map(chembl2uni)

df_annot_prot = df_chembl[['UniprotID', 'l1']].dropna().drop_duplicates()

# remove repeated with unclassified

df_morethan1 = df_annot_prot[df_annot_prot.UniprotID.duplicated()].sort_values(by='UniprotID')
more_than_1_annot = df_morethan1.UniprotID.unique().tolist()

drop_index_not_annot = []

for protein in more_than_1_annot:
    indexes = df_annot_prot[df_annot_prot.UniprotID == protein].index.tolist()
    for idx in indexes:
        val = df_annot_prot.loc[idx].l1
        if val == 'Unclassified protein':
            drop_index_not_annot.append(idx)

df_annot_prot = df_annot_prot.drop(index=drop_index_not_annot)

# the rest we will keep first
df_annot_prot = df_annot_prot.drop_duplicates(subset = 'UniprotID', keep='first')

assert df_annot_prot.UniprotID.duplicated().any() == False, 'error'

print(f'There are {len(df_annot_prot.l1.unique())} unique protein classes')

protein_fam_class = dict(zip(df_annot_prot.UniprotID.tolist(), df_annot_prot.l1.tolist()))


protein_fam_class


PATH_OUT_DICT = 'Results/6th_tsne/annotation/'

with open(PATH_OUT_DICT+'protein_fam_class.json', 'w') as f:
    json.dump(protein_fam_class, f)




### GET ANNOT ENZYMES specifc.

df_annot_enzymes = df_chembl[['UniprotID', 'l1', 'l2']].dropna().drop_duplicates()
df_annot_enzymes = df_annot_enzymes[df_annot_enzymes.l1 == 'Enzyme'].drop(columns='l1')

df_annot_enzymes[df_annot_enzymes.UniprotID.duplicated()].shape[0]

df_annot_enzymes = df_annot_enzymes.drop_duplicates(subset = 'UniprotID', keep='first')


enzymes_annot = dict(zip(df_annot_enzymes.UniprotID.tolist(), df_annot_enzymes.l2.tolist()))

enzymes_annot

with open(PATH_OUT_DICT+'enzymes_annot.json', 'w') as f:
    json.dump(enzymes_annot, f)

