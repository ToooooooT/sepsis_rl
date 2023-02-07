######################################################################################
# Import package
######################################################################################

import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import gc

source_path = '../data/mimiciii/source_data/'
dest_path = '../data/mimiciii/preprocess_data/'

######################################################################################
# Split CHARTEVENTS to several dataset
######################################################################################
print('start splitting CHARTEVENTS to 9 chunck...')
count = 1
for reader in tqdm(pd.read_csv(source_path + 'CHARTEVENTS.csv', chunksize = 40000000)):
    reader.to_csv(dest_path + 'CHARTEVENTS' + str(count) + '.csv', index=False)
    count += 1
print('finish splitting')

######################################################################################
# Load Data
######################################################################################

paths = sorted(os.listdir(source_path))
remove_paths = list()
for path in paths:
    if path[-4:] != '.csv':
        remove_paths.append(path)
for path in remove_paths:
    paths.remove(path)
paths.remove('CALLOUT.csv')
paths.remove('CAREGIVERS.csv')
paths.remove('CHARTEVENTS.csv')
paths.remove('CPTEVENTS.csv')
paths.remove('DATETIMEEVENTS.csv')
paths.remove('SERVICES.csv',)
paths.remove('TRANSFERS.csv')
paths.remove('D_CPT.csv')

datas = list()
for path in paths:
    datas.append(pd.read_csv(source_path + path))
    
for path in sorted(os.listdir(dest_path)):
    datas.append(pd.read_csv(dest_path + path))

paths += sorted(list(os.listdir(dest_path)))

######################################################################################
# Clean Data to remain SEPSIS patient and sort Data
######################################################################################
print('start filter patient...')
# filter diagnoses to sepsis
i = paths.index('D_ICD_DIAGNOSES.csv')
D_ICD_DIAGNOSES = datas[i].copy()
D_ICD_DIAGNOSES.drop(D_ICD_DIAGNOSES[['sepsis' not in s.lower() for s in D_ICD_DIAGNOSES['LONG_TITLE']]].index, inplace=True)
D_ICD_DIAGNOSES.to_csv(dest_path + paths[i], index=False)

# filter subject_id
i = paths.index('DIAGNOSES_ICD.csv')
datas[i].sort_values(by=['SUBJECT_ID', 'SEQ_NUM'], inplace=True)
DIAGNOSES_ICD = datas[i].copy()

patient = set()
for j in tqdm(DIAGNOSES_ICD.index):
    for code in D_ICD_DIAGNOSES['ICD9_CODE']:
        if str(DIAGNOSES_ICD.loc[j]['ICD9_CODE']) == code:
            patient.add((DIAGNOSES_ICD.loc[j]['SUBJECT_ID'], DIAGNOSES_ICD.loc[j]['HADM_ID']))
            break

drop_index = list()
for j in tqdm(DIAGNOSES_ICD.index):
    subject_id = DIAGNOSES_ICD.loc[j, 'SUBJECT_ID']
    hadm_id = DIAGNOSES_ICD.loc[j, 'HADM_ID']
    if (subject_id, hadm_id) not in patient:
        drop_index.append(j)
DIAGNOSES_ICD.drop(drop_index, inplace=True)
DIAGNOSES_ICD.to_csv(dest_path + paths[i], index=False)

subject_id, hadm_id = list(), list()
for sub, hadm in patient:
    subject_id.append(sub)
    hadm_id.append(hadm)
dataset = pd.DataFrame({'SUBJECT_ID' : subject_id, 'HADM_ID' : hadm_id})
dataset.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
dataset.index = range(len(dataset))
dataset.to_csv('../data/mimiciii/patient.csv', index=False)

dict1 = defaultdict(list)
for i in range(len(dataset)):
    dict1[dataset['SUBJECT_ID'][i]].append(dataset['HADM_ID'][i])

for i in tqdm(range(len(datas))):
    if i == paths.index('DIAGNOSES_ICD.csv'):
        continue
    if 'SUBJECT_ID' in datas[i].columns and 'HADM_ID' in datas[i].columns:
        drop_index = list()
        for j in datas[i].index:
            if datas[i]['HADM_ID'][j] not in dict1[datas[i]['SUBJECT_ID'][j]]:
                drop_index.append(j)
        datas[i].drop(drop_index, inplace=True)
        del drop_index
        # datas[i].sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    elif 'SUBJECT_ID' in datas[i].columns:
        datas[i] = datas[i][[id in dataset['SUBJECT_ID'].values for id in datas[i]['SUBJECT_ID']]]
        # datas[i].sort_values(by=['SUBJECT_ID'], inplace=True)

    datas[i].to_csv(dest_path + paths[i], index=False)
    
del dict1
gc.collect()
        
i = paths.index('D_ICD_PROCEDURES.csv')
datas[i].sort_values(by=['ICD9_CODE'], inplace=True)
datas[i].to_csv(dest_path + paths[i], index=False)

i = paths.index('D_ITEMS.csv')
datas[i].sort_values(by=['ITEMID'], inplace=True)
datas[i].to_csv(dest_path + paths[i], index=False)

i = paths.index('D_LABITEMS.csv')
datas[i].sort_values(by=['ITEMID'], inplace=True)
datas[i].to_csv(dest_path + paths[i], index=False)

print('finish filter patient')