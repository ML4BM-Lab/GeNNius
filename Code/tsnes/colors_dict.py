

def return_colordic():

    dict_protfam = {'Enzyme': '#1F77B4', 
                    'Membrane receptor': '#FF7F0E', 
                    'Transporter': '#2CA02C', 
                    'Secreted protein': '#D62728', 
                    'Ion channel': '#9467BD', 
                    'Transcription factor': '#8C564B', 
                    'Auxiliary transport protein': '#E377C2', 
                    'Epigenetic regulator': '#7F7F7F', 
                    'Structural protein': '#BCBD22', 
                    'Other cytosolic protein': '#17BECF',
                    'Other': '#778AAE', 
                    'Unclassified protein':'#222A2A'
                    }

    dict_enz = {'Protease': '#2E91E5', 
                'Isomerase': '#1CA71C', 
                'Cytochrome P450': '#FB0D0D', 
                'Oxidoreductase': '#780000', # change this one, rly close to  kinase
                'Lyase': '#222A2A', 
                'Hydrolase': '#B68100', 
                'Transferase': '#750D86', 
                'Ligase': '#EB663B', 
                'Phosphatase': '#511CFB', 
                'Phosphodiesterase': '#00A08B', 
                'Kinase': '#FB00D1', 
                'Aminoacyltransferase': '#8B8000' # changed this one
                }


    dict_drugfamcols = {'Organic Polymers': '#2E91E5', 
                        'Organic acids and derivatives': '#1CA71C', 
                        'Organoheterocyclic compounds': '#FB0D0D', 
                        'Nucleosides, nucleotides, and analogues': '#B68100', 
                        'Organic nitrogen compounds': '#DA16FF', 
                        'Lipids and lipid-like molecules': '#750D86', 
                        'Organic oxygen compounds':  '#EB663B', 
                        'Benzenoids': '#511CFB', 
                        'Phenylpropanoids and polyketides': '#00A08B', 
                        'Alkaloids and derivatives': '#DA60CA',  #'#FB00D1'
                        'Mixed metal/non-metal compounds': '#862A16',
                        'Organosulfur compounds': '#A777F1',
                        'Other': '#778AAE',
                        'Unclassified': '#222A2A'
                        }
    return dict_protfam, dict_enz, dict_drugfamcols