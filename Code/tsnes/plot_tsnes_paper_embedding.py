# 
import os
import numpy as np 
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from Code.tsnes.colors_dict import return_colordic

########
dict_protfam, dict_enz, dict_drugfamcols = return_colordic()

DPI = 400
########

FOLDER_ANNOT = 'Annotation'

df_annots_drugs = pd.read_pickle(os.path.join(FOLDER_ANNOT, 'df_annots_drugs.pkl'))
df_annots_prots = pd.read_pickle(os.path.join(FOLDER_ANNOT, 'df_annots_prots.pkl'))
df_annots_edges = pd.read_pickle(os.path.join(FOLDER_ANNOT, 'df_annots_edges.pkl'))

#
DATABASE = 'DrugBank'.lower()
hd = 17
DATA_PATH =  'Data' #os.path.join('Data', f'{DATABASE.upper()}' )

data = torch.load(os.path.join(DATA_PATH, f'hetero_data_{DATABASE}.pt')).to('cpu')

edge_idx = data['drug','protein'].edge_index.numpy()
list_drug_idx = edge_idx[:][0] # for colors
list_protein_idx = edge_idx[:][1] # for colors

EMB_PATH = f'Embeddings_{hd}' #os.path.join('Results', '1st_section_results', f'{DATABASE.upper()}_{hd}' )
emb_drug = np.load(f'{EMB_PATH}/embeddings_drugs.npy')
emb_prot = np.load(f'{EMB_PATH}/embeddings_proteins.npy')



matrix_concat = []

# create concat matrix
for idx in range(edge_idx.shape[1]):
    idx_drug = list_drug_idx[idx]
    idx_prot = list_protein_idx[idx]
    cat = np.concatenate((emb_drug[idx_drug], emb_prot[idx_prot]), axis=0)#.shape
    matrix_concat.append(cat)

matrix_concat = np.array(matrix_concat)

assert matrix_concat.shape == (edge_idx.shape[1], hd*2), 'shapes'

####

OUT_FOLDER = 'Results' #6th_tsne/dimensionality_red_innit'


# tsne ags (USED)
# perp = 250 #
# niter= 1_500
# met = 'cosine' 
# init = 'random' 

perp = 50 #
niter= 1_500
met = 'cosine' 
init = 'random' 

################ DRUGS PLOT
data_type = 'drugfam'
#features = emb_drug
dict4colors = dict_drugfamcols
color_drugs = df_annots_drugs.drugfam_filtered.map(dict_drugfamcols).tolist()

indices_drug = [i for i, x in enumerate(df_annots_drugs.drugfam_filtered.tolist()) if x not in ['Unclassified']]
del dict4colors["Unclassified"]

tsne_ = TSNE(random_state=42, init=init, perplexity=perp, n_iter=niter, metric=met)
tsne = tsne_.fit_transform(emb_drug)

# plot
plt.clf()
plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_facecolor("white")
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
plt.scatter(tsne.T[0], tsne.T[1], s=12.5, alpha=0.65,  marker="o", c=color_drugs) 
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dict4colors.values()]
plt.legend(markers, dict4colors.keys(), numpoints=1, fontsize=8, loc='lower right', ncol=2).get_frame().set_alpha(0.7)
out_fig = os.path.join(OUT_FOLDER, f'tsne_embedding_{data_type}.pdf')
plt.savefig(out_fig, dpi=DPI, bbox_inches='tight',  pad_inches = 0.2)


# opts_drugs = ['drug_degree', 'drug_degree_centrality', 'drug_closeness_centrality', 'drug_betweenness_centrality', 'drug_subgraph_centrality', 'annot_drugs_comp']

# for opt in opts_drugs:
#     color_drugs = df_annots_drugs[opt].tolist()
#     out_fig = os.path.join(OUT_FOLDER, f'tsne_{data_type}_{opt}.png')
#     plt.clf()
#     plt.figure(figsize=(10,10))
#     ax = plt.axes()
#     ax.set_facecolor("white")
#     plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
#     plt.scatter(tsne.T[0], tsne.T[1], s=9.5, alpha=0.75,  marker="o", c=color_drugs, cmap='Oranges') 
#     #markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dict4colors.values()]
#     #plt.legend(markers, dict4colors.keys(), numpoints=1, fontsize=8, loc='upper left').get_frame().set_alpha(0.5)
#     plt.savefig(out_fig, dpi=400, bbox_inches='tight',  pad_inches = 0.2)


######## PROTEIN PLOT

data_type = 'protfam'

dict4colors = dict_protfam
color_prots = df_annots_prots.protfam_filtered.map(dict_protfam).tolist()

### TSNE
tsne_ = TSNE(random_state=42, init=init, perplexity=perp, n_iter=niter, metric=met)
tsne = tsne_.fit_transform(emb_prot)

indices_prot = [i for i, x in enumerate(df_annots_prots.protfam_filtered.tolist()) if x not in ['Unclassified protein']]
del dict4colors['Unclassified protein']

plt.clf()
plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_facecolor("white")
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
plt.scatter(tsne.T[0][indices_prot], tsne.T[1][indices_prot], s=20.5, alpha=0.65,  marker="o", c=np.array(color_prots)[indices_prot]) 
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dict4colors.values()]
plt.legend(markers, dict4colors.keys(), numpoints=1, fontsize=11, loc='lower right', ncol=2).get_frame().set_alpha(0.7)
out_fig = os.path.join(OUT_FOLDER, f'tsne_embedding_{data_type}.pdf')
plt.savefig(out_fig, dpi=DPI, bbox_inches='tight',  pad_inches = 0.2)


### enzymes
data_type = 'enzfam'
dict4colors = dict_enz
color_enz = df_annots_prots.protenz.map(dict_enz).tolist()
indices_enz = [i for i, x in enumerate(df_annots_prots.protenz.tolist()) if x not in ['None']]

plt.clf()
plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_facecolor("white")
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
plt.scatter(tsne.T[0][indices_enz], tsne.T[1][indices_enz], s=20.5, alpha=0.65,  marker="o", c=np.array(color_enz)[indices_enz]) 
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dict4colors.values()]
plt.legend(markers, dict4colors.keys(), numpoints=1, fontsize=14, loc='lower right', ncol=2).get_frame().set_alpha(0.7)
out_fig = os.path.join(OUT_FOLDER, f'tsne_embedding_{data_type}.pdf')
plt.savefig(out_fig, dpi=DPI, bbox_inches='tight',  pad_inches = 0.2)


# opts_prots = ['protein_degree', 'protein_degree_centrality', 'protein_closeness_centrality', 'protein_betweenness_centrality', 'protein_subgraph_centrality', 'annot_proteins_connected']

# for opt in opts_prots:
#     color_prots = df_annots_prots[opt].tolist()
#     out_fig = os.path.join(OUT_FOLDER, f'tsne_{data_type}_{opt}.png')
#     plt.clf()
#     plt.figure(figsize=(10,10))
#     ax = plt.axes()
#     ax.set_facecolor("white")
#     plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
#     plt.scatter(tsne.T[0], tsne.T[1], s=9.5, alpha=0.75,  marker="o", c=color_prots, cmap='Oranges') 
#     plt.savefig(out_fig, dpi=400, bbox_inches='tight',  pad_inches = 0.2)


####### EDGESS !
# tsne ags (USED)
perp = 30 #
niter= 1_000 #
met = 'cosine' # 
init = 'random' #

### TSNE
tsne_ = TSNE(random_state=1, init=init, perplexity=perp, n_iter=niter, metric=met)
tsne = tsne_.fit_transform(matrix_concat)

########### EDGE DRUGFAM
data_type = 'edges_drugfam'

dict4colors = dict_drugfamcols
color_drugs = df_annots_edges.drugfam_filtered_edges.map(dict_drugfamcols).tolist()
indices_drug = [i for i, x in enumerate(df_annots_edges.drugfam_filtered_edges.tolist()) if x not in ['Unclassified']]
del dict4colors["Unclassified"]

plt.clf()
plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_facecolor("white")
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
#plt.scatter(tsne.T[0], tsne.T[1], s=9.5, alpha=0.75,  marker="o", c=color_drugs) 
plt.scatter(tsne.T[0][indices_drug], tsne.T[1][indices_drug], s=10.5, alpha=0.6,  marker="o", c=np.array(color_drugs)[indices_drug]) 
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dict4colors.values()]
plt.legend(markers, dict4colors.keys(), numpoints=1, fontsize=10, loc='upper left', ncol=2).get_frame().set_alpha(0.7)
out_fig = os.path.join(OUT_FOLDER, f'tsne_embedding_{data_type}.pdf')
plt.savefig(out_fig, dpi=DPI, bbox_inches='tight',  pad_inches = 0.2)


########### EDGE PROT
data_type = 'edges_protfam'

dict4colors = dict_protfam
color_prots = df_annots_edges.protfam_filtered_edges.map(dict4colors).tolist()

# check numbers may not be ok
indices_prot = [i for i, x in enumerate(df_annots_edges.protfam_filtered_edges.tolist()) if x not in ['Unclassified protein']]
del dict4colors['Unclassified protein']


plt.clf()
plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_facecolor("white")
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
#plt.scatter(tsne.T[0], tsne.T[1], s=10.5, alpha=0.75,  marker="o", c=color_prots) 
plt.scatter(tsne.T[0][indices_prot], tsne.T[1][indices_prot], s=10.5, alpha=0.7,  marker="o", c=np.array(color_prots)[indices_prot]) 
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dict4colors.values()]
plt.legend(markers, dict4colors.keys(), numpoints=1, fontsize=11, loc='upper left').get_frame().set_alpha(0.7)
out_fig = os.path.join(OUT_FOLDER, f'tsne_embedding_{data_type}.pdf')
plt.savefig(out_fig, dpi=DPI, bbox_inches='tight',  pad_inches = 0.2)


########### EDGE ENZYMES
data_type = 'edges_enz'

dict4colors = dict_enz
color_prots = df_annots_edges.protenz_edges.map(dict4colors).tolist()

indices_enz = [i for i, x in enumerate(df_annots_edges.protenz_edges.tolist()) if x not in ['None']]

plt.clf()
plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_facecolor("white")
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
plt.scatter(tsne.T[0][indices_enz], tsne.T[1][indices_enz], s=10.5, alpha=0.75,  marker="o", c=np.array(color_prots)[indices_enz]) 
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dict4colors.values()]
plt.legend(markers, dict4colors.keys(), numpoints=1, fontsize=11, loc='upper left', ncol=2).get_frame().set_alpha(0.7)
out_fig = os.path.join(OUT_FOLDER, f'tsne_embedding_{data_type}.pdf')
plt.savefig(out_fig, dpi=DPI, bbox_inches='tight',  pad_inches = 0.2)


# opts_edges = ['drug_degree_edges', 'drug_degree_centrality_edges', 'drug_closeness_centrality_edges', 
#  'drug_betweenness_centrality_edges', 'drug_subgraph_centrality_edges', 'protein_degree_edges',
#    'protein_degree_centrality_edges', 'protein_closeness_centrality_edges', 
#    'protein_betweenness_centrality_edges', 'protein_subgraph_centrality_edges']

# for opt in opts_edges:
#     color_prots = df_annots_edges[opt].tolist()
#     out_fig = os.path.join(OUT_FOLDER, f'tsne_{data_type}_{opt}.png')
#     plt.clf()
#     plt.figure(figsize=(10,10))
#     ax = plt.axes()
#     ax.set_facecolor("white")
#     plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
#     plt.scatter(tsne.T[0], tsne.T[1], s=9.5, alpha=0.75,  marker="o", c=color_prots, cmap='Oranges') 
#     plt.savefig(out_fig, dpi=400, bbox_inches='tight',  pad_inches = 0.2)
