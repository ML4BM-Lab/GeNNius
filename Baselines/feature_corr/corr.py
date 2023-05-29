import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

names_drug = [
        'MolLogP',
        'MolWt',
        'NumHAcceptors',
        'NumHDonors',
        'NumHeteroatoms',
        'NumRotatableBonds',
        'TPSA',
        'RingCount',
        'NHOHCount',
        'NOCount',
        'HeavyAtomCount',
        'NumValenceElectrons',
        ]

list_aminos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 
                'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 
                'T', 'W', 'Y', 'V']

list_aminos.sort()

###

# for database in ['e', 'DrugBank']:
#     print(database)
#     for data_type in ['drug', 'protein']:

database = 'e'
data_type = 'protein'
print(data_type)

#PATH_DRUGBANK_GENNIUS = '../GENNIUS/Data/DRUGBANK/hetero_data_drugbank.pt'
#PATH_E_GENNIUS = '../GENNIUS/Data/E/hetero_data_e.pt'

path_data_gennius = f'../GENNIUS/Data/{database.upper()}/hetero_data_{database.lower()}.pt'

data = torch.load(path_data_gennius)

fts = data[data_type].x.cpu().numpy()

df = pd.DataFrame(fts)
corr = df.corr()

if data_type == 'drug':
    corr.columns = names_drug

else:
    corr.columns = list_aminos

plt.clf()
#plt.figure(figsize=(10,10))
f, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(corr,
    cmap=sns.diverging_palette(220, 10, sep=50, as_cmap=True), # 100 dor grus is ok
    vmin=-1.0, vmax=1.0,
    square=True,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    ax=ax,
    cbar_kws={"shrink": 0.85}) 

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig(f'Figures/corr_{data_type}_{database}.pdf', dpi=330)
