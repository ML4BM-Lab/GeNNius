"""
This code executes preprocessing for all combination of databases,
needed to reproduce results of generalization section. 

"""

import logging
import subprocess as sp


datasets_o = ['DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR'] 

datasets = datasets_o

for database1 in datasets_o:
    for database2 in datasets:
        if database2 != database1:
            try:
                print(f'database 1 {database1}, database 2 {database2}')
                return_code = sp.check_call(['python3', 'Code/data_preprocessing_removing_repeated_edges.py', '-d', f'{database1}', '-t', f'{database2}'])
                if return_code ==0: 
                    logging.info(f'EXIT CODE 0 FOR {database1.upper()} w\o {database2.upper()}')

            except sp.CalledProcessError as e:
                logging.info(e.output)


print('0')