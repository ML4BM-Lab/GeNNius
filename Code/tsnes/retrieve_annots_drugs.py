####################################
################## DRUGS
import os
import pandas as pd
from tqdm import tqdm
import logging
import json



#import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
from tqdm import tqdm
import time

def get_drug2superclass(inkey, sleep=5):
    """
    Drug Superclass
    """

    #inkey = dict_pub2key.get(drugid)
    superclass = None
    # http session stuff
    retry_strategy = Retry(
    total=3,
    #status_forcelist = [429], # , 500, 502, 503, 504
    method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # mount session
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    try:
        response = http.get(f'http://classyfire.wishartlab.com/entities/{inkey}.json')
    except:
        logging.info(f'Connection did not work {sys.exc_info()[0]}')

    if response.status_code == 200:
        json = response.json()
        if json.get('superclass', {}).get('name'):
            superclass = json.get('superclass').get('name') 
            #logging.debug(f'worked: {superclass}')
            print(f'worked: {superclass}')

        time.sleep(sleep)
    else:
        logging.info(f'status code {response.status_code} for {inkey}')
    return superclass



def pubchem2inchikey_batch(drugs, size=200):
    pub2key = {}

    # split the list in chunks of 100 to make requests
    drugs = [str(drug) for drug in drugs]
    split_list = lambda big_list, x: [
        big_list[i : i + x] for i in range(0, len(big_list), x)
    ]

    drug_chunks = split_list(drugs, size)
    for chunk in tqdm(drug_chunks, desc='Requesting SMILES to PubChem'):
        chunk = ",".join(chunk)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{chunk}/json"
        response = requests.get(url)
        if response.status_code == 200:
            jsons = response.json()
            for id in jsons.get("PC_Compounds"):
                #
                cid, smile, inchikey = None, None, None
                cid = id.get('id').get('id').get('cid')
                cid = str(cid)

                inchikey =  [prop.get('value').get('sval') for prop in id.get('props') 
                                if prop.get('urn').get('label') == 'InChIKey'][0]


                pub2key[cid] = inchikey
                #res.append((cid, inchikey))
    return pub2key


logging.basicConfig(level=logging.INFO)

DATABASE = 'DrugBank'.lower()

DATA_PATH = os.path.join('Data', DATABASE.upper())

with open(os.path.join(DATA_PATH, 'drug_mapping.json'), 'r') as f:
  drug_mapping = json.load(f)

drugs_unique = list(drug_mapping.keys())
dict_pub2key = pubchem2inchikey_batch(drugs_unique)

drug2key = {key: dict_pub2key.get(key, None) for key in drugs_unique
                    if dict_pub2key.get(key, None)}

drug2superclass = {drug: get_drug2superclass(key, sleep=8) for drug, key in drug2key.items()}


print(drug2superclass)

with open(f'Results/drug2superclass_{DATABASE.lower()}.json', "w") as outfile:
    json.dump(drug2superclass, outfile)


print('FINISHED')