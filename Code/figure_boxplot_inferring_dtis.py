import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# path_files = 'Results/inferring_dtis/'

# fileslist = glob.glob(path_files + '*_statistics_results.txt')

# fileslist[0]
# datasets = [file.replace(path_files,'').replace('_statistics_results.txt', '') for file in fileslist]


datasets = ['DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']
annots_nedges = [4604, 4765, 461, 24, 338, 162, 292, 10]
annots_nedges = [str(value) for value in annots_nedges]

x_ticks = [datasets[i] +'\n' + 'e=' + annots_nedges[i] for i in range(len(annots_nedges))]


#dataset = datasets[0]

results = pd.DataFrame(columns=['dataset', 'th'])

for dataset in datasets:
    df = pd.read_csv(f'Results/inferring_dtis/{dataset.lower()}_statistics_results.txt', sep="\t", names=['th'])
    #df.columns = ['th05']
    df['dataset'] = dataset
    results = pd.concat([results, df])

results = results.reset_index().drop(columns='index')


plt.clf()
plt.figure(figsize=(18, 10))
#boxplot = 
boxplot = sns.boxplot(data=results, x="dataset", y="th", palette='flare')#, hue="thnames")
boxplot.set_xlabel( "Datasets", fontsize = 24)
boxplot.set_ylabel( "Ratio of positive edges detected", fontsize = 24)
boxplot.set_xticklabels(x_ticks, fontsize = 20)
boxplot.set_yticklabels([str(round(i,1)) for i in boxplot.get_yticks()], fontsize = 20)
#plt.savefig('Results/inferring_dtis/boxplot_10runs.pdf', dpi=330 ,bbox_inches='tight',  pad_inches = 0.25)
plt.savefig('Results/inferring_dtis/boxplot_validation.pdf', dpi=330 ,bbox_inches='tight',  pad_inches = 0.25)

exit(0)

