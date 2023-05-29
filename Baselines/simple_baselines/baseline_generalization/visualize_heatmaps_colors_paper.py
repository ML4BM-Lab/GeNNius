import os
import pandas as pd
import matplotlib.pyplot as plt
#from Code.utils import plot_heatmap
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def plot_heatmap(df,df_std, cmap='Reds'):
    # Create an array to annotate the heatmap
    labels = ["{0:.4f}\n$\pm$\n{1:.4f}".format(symb,value) for symb, value in zip(df.values.flatten(), df_std.values.flatten())]
    labels = np.asarray(labels).reshape(df.shape)
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = sns.heatmap(df, annot=labels, fmt="", vmin=0.50, vmax=1.0, cmap=cmap)
    ax.set(xlabel="Trained", ylabel="Evaluated")
    ax.xaxis.tick_top()


# hidden_channels = 17
# PATH = 'Results/2nd_section_gen_with_17_fromGS'

df_auc_hmp = pd.read_pickle(f'auc_mean.pkl')
df_aupr_hmp = pd.read_pickle( f'aupr_mean.pkl')
df_auc_std_hmp = pd.read_pickle(f'auc_std.pkl')
df_aupr_std_hmp = pd.read_pickle(f'aupr_std.pkl')




cvals  = [0, 1]
colors = ["white", "#0f2057"]

norm = plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), colors))
cmap = LinearSegmentedColormap.from_list("", tuples)

plot_heatmap(df_auc_hmp, df_auc_std_hmp, cmap=cmap)
plt.savefig(os.path.join(f'baseline_RF_results_AUC_17_mod.pdf') ,bbox_inches='tight',  pad_inches = 0.2)



cvals  = [0, 1]
colors = ["white", "#520035"]

norm = plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), colors))
cmap = LinearSegmentedColormap.from_list("", tuples)

plot_heatmap(df_aupr_hmp, df_aupr_std_hmp, cmap)
plt.savefig(os.path.join(f'baseline_RF_results_AUPR_17_mod.pdf'), bbox_inches='tight',  pad_inches = 0.2)
