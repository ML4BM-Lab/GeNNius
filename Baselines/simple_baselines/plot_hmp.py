import pandas as pd
from Code.utils import plot_heatmap
import matplotlib.pyplot as plt


PATH = 'Baselines/baseline_generalization/'
# save data
df_auc_hmp = pd.read_pickle(PATH + f'auc_mean.pkl')
df_aupr_hmp = pd.read_pickle(PATH + f'aupr_mean.pkl')
df_auc_std_hmp = pd.read_pickle(PATH + f'auc_std.pkl')
df_aupr_std_hmp = pd.read_pickle(PATH +f'aupr_std.pkl')


## PLOT HEATMAP AUC
plot_heatmap(df_auc_hmp, df_auc_std_hmp, cmap='Purples')
plt.savefig(PATH + f'results_RF_AUC_purples.pdf',bbox_inches='tight',  pad_inches = 0.2)

## PLOT HEATMAP AUPR
plot_heatmap(df_aupr_hmp, df_aupr_std_hmp, cmap='Purples')
plt.savefig(PATH + f'results_RF_AUPR_purples.pdf',bbox_inches='tight',  pad_inches = 0.2)