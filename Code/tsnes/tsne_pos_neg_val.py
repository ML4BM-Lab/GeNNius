# MODIFIED VERSION ONLY READING FILES

# working here (11Aprl)
import os
import numpy as np 
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import json

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

#from Code.tsnes.colors_dict import return_colordic

#dict_protfam, dict_enz, dict_drugfamcols = return_colordic()


OUT_FOLDER_READ = 'Annotation'

df_annots_drugs = pd.read_pickle(os.path.join(OUT_FOLDER_READ, 'df_annots_drugs.pkl'))
df_annots_prots = pd.read_pickle(os.path.join(OUT_FOLDER_READ, 'df_annots_prots.pkl'))
df_annots_edges = pd.read_pickle(os.path.join(OUT_FOLDER_READ, 'df_annots_edges.pkl'))

#
dataset = 'drugbank'.lower()

hd = 17
#device = 'cuda'


OUT_TSNE = 'Pos_Neg'

tsne = np.load(os.path.join(OUT_TSNE, 'tsne_pos_neg_val.npy'))

# save for annotation
edge_idx = np.load(os.path.join(OUT_TSNE, 'edge_idx_train.npy'))
edge_idx_val = np.load(os.path.join(OUT_TSNE, 'edge_idx_val.npy'))

# and the matrix
matrix_concat_train = np.load(os.path.join(OUT_TSNE, 'matrix_concat_train.npy'))
matrix_concat_val = np.load (os.path.join(OUT_TSNE, 'matrix_concat_val.npy'))

# save values
train_data_edge_labels = np.load(os.path.join(OUT_TSNE, 'train_data_edge_labels.npy'))
val_out_labels = np.load(os.path.join(OUT_TSNE, 'val_out_labels.npy'))


tsne_train = tsne[:matrix_concat_train.shape[0]]
assert tsne_train.shape[0] == matrix_concat_train.shape[0], 'shapes fail'
tsne_val = tsne[matrix_concat_train.shape[0]:]
assert tsne_val.shape[0] == matrix_concat_val.shape[0]



# Pastel Blue: #AED6F1
# Dark Blue: #3498DB
# Pastel Red: #F1948A
# Dark Red: #C0392B

dict4colors = {'Positive in DrugBank': '#F1948A',
               'Negative in DrugBank': '#AED6F1',
               'Validation predicted as positive': '#C0392B',
               'Validation predicted as negative': '#3498DB'}


colors_train = ['#F1948A' if item == 1 else '#AED6F1' for item in train_data_edge_labels]
colors_val = ['#C0392B' if item>=0.5 else '#3498DB' for item in val_out_labels]

plt.clf()
plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_facecolor("white")
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
plt.scatter(tsne_train.T[0], tsne_train.T[1], s=13, alpha=0.9,  marker="o", c=colors_train) 
plt.scatter(tsne_val.T[0], tsne_val.T[1], s=13.5, alpha=0.6,  marker="p", c=colors_val)  # out_val.cpu().tolist(), cmap='Greys'
markers = [plt.Line2D([0,0],[0,0], color=color, marker=marker, linestyle='') for color, marker in zip(dict4colors.values(), ['o','o', 'p', 'p'])]
plt.legend(markers, dict4colors.keys(), numpoints=1, fontsize=14, loc='lower right', ncol=1).get_frame().set_alpha(0.7)
out_fig = os.path.join('Results', 'edges_embedded_space_val_new.pdf')#os.path.join(OUT_FOLDER, f'tsne_{data_type}_fams.png')
plt.savefig(out_fig, dpi=400, bbox_inches='tight',  pad_inches = 0.2)





plt.clf()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

dict4colors = {'Positive': '#F1948A',
               'Negative': '#AED6F1',
               r'$(+)$ predicted as $(+)$': '#C0392B',
               r'$(+)$ predicted as $(-)$': '#3498DB'}

# Plot data on first subplot
ax1.scatter(tsne_train.T[0], tsne_train.T[1], s=13, alpha=0.9,  marker="o", c=colors_train) 
ax1.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
markers = [plt.Line2D([0,0],[0,0], color=color, marker=marker, linestyle='') for color, marker in zip(dict4colors.values(), ['o','o', 'p', 'p'])]
ax1.scatter(tsne_train.T[0], tsne_train.T[1], s=13, alpha=0.9,  marker="o", c=colors_train) 
ax1.scatter(tsne_val.T[0], tsne_val.T[1], s=13.5, alpha=0.6,  marker="p", c=colors_val)
markers = [plt.Line2D([0,0],[0,0], color=color, marker=marker, linestyle='') for color, marker in zip(dict4colors.values(), ['o','o', 'p', 'p'])]
ax1.legend(markers, dict4colors.keys(), numpoints=1, fontsize=14, loc='upper left', ncol=2).get_frame().set_alpha(0.7)
# ax1.legend()
# Plot data on second subplot

ax2.scatter(tsne_train.T[0], tsne_train.T[1], s=13, alpha=0.9,  marker="o", c=colors_train) 
ax2.scatter(tsne_val.T[0], tsne_val.T[1], s=13.5, alpha=0.95,  marker="p", c=val_out_labels, cmap='Greys')# ax1.legend()
ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
# ax1.legend()# Remove space between subplots
fig.subplots_adjust(wspace=0.025)
out_fig = os.path.join('Results', 'edges_embedded_space_val_new_2.pdf')
plt.savefig(out_fig, dpi=350, bbox_inches='tight',  pad_inches = 0.1)



###

plt.clf()

import matplotlib.pyplot as plt
import matplotlib as mpl

fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap='Greys')


plt.savefig("Results/colorbar.pdf", format='pdf',  bbox_inches='tight')
