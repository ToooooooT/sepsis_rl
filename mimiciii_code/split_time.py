######################################################################################
# Import package
######################################################################################

import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

pd.options.mode.chained_assignment = None

preprocess_path = '../data/mimiciii/preprocess_data/'
temporal_path = '../data/mimiciii/temporal_dataset/'

######################################################################################
# Load Data
######################################################################################
print('loading data...')

datas = list()
paths = sorted(os.listdir(preprocess_path))
for path in paths:
    datas.append(pd.read_csv(preprocess_path + path))

print('finish loading data')

######################################################################################
# Look Data characteristics
######################################################################################

for i in range(len(datas)):
    for column in datas[i].columns:
        if 'TIME' in column or 'DATE' in column or 'DOB' == column or 'DOD' == column or 'DOD_HOSP' == column or 'DOD_SSN' == column:
            datas[i][column] = datas[i][column].apply(lambda x : pd.Timestamp(x))

i = paths.index('D_ICD_DIAGNOSES.csv')
D_ICD_DIAGNOSES = datas[i].copy()

i = paths.index('DIAGNOSES_ICD.csv')
DIAGNOSES_ICD = datas[i].copy()

i = paths.index('ICUSTAYS.csv')
ICUSTAYS = datas[i].copy()

i = paths.index('PATIENTS.csv')
PATIENTS = datas[i].copy()

i = paths.index('ADMISSIONS.csv')
ADMISSIONS = datas[i].copy()

######################################################################################
# Load dataset
######################################################################################

dataset = pd.read_csv('../data/mimiciii/patient.csv')

######################################################################################
# Add gender and age to dataset
######################################################################################

dataset['DOB'] = np.nan
dataset['Gender'] = np.nan

for i in PATIENTS.index:
    subject_id, gender, dob = PATIENTS.loc[i, ['SUBJECT_ID', 'GENDER', 'DOB']]
    index = dataset[dataset['SUBJECT_ID'] == subject_id].index
    dataset['DOB'].loc[index] = dob
    dataset['Gender'].loc[index] = gender
    
dataset['Gender'] = dataset['Gender'].apply(lambda x : 1 if x == 'M' else 0)

######################################################################################
# Add death time, admit time, discharge time to dataset
######################################################################################

dataset['DEATHTIME'] = np.nan
dataset['ADMITTIME'] = np.nan
dataset['DISCHTIME'] = np.nan

for i in ADMISSIONS.index:
    subject_id, hadm_id, death_time, admittime, dischtime = ADMISSIONS.loc[i, ['SUBJECT_ID', 'HADM_ID', 'DEATHTIME', 'ADMITTIME', 'DISCHTIME']]
    index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index
    dataset['DEATHTIME'].loc[index] = death_time
    dataset['ADMITTIME'].loc[index] = admittime
    dataset['DISCHTIME'].loc[index] = dischtime

######################################################################################
# change time by ICUSTAY_ID in dataset
######################################################################################

for i in ICUSTAYS.index:
    subject_id, hadm_id, intime, outtime = ICUSTAYS.loc[i, ['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME']]
    index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index
    if dataset['ADMITTIME'].isnull()[index[0]]:
        dataset['ADMITTIME'].loc[index[0]] = intime
    elif dataset['ADMITTIME'].loc[index[0]] - intime > pd.Timedelta('00:00:00'):
        dataset['ADMITTIME'].loc[index[0]] = intime
    if dataset['DISCHTIME'].isnull()[index[0]]:
        dataset['DISCHTIME'].loc[index[0]] = outtime
    elif dataset['DISCHTIME'].loc[index[0]] - outtime < pd.Timedelta('00:00:00'):
        dataset['DISCHTIME'].loc[index[0]] = outtime

######################################################################################
# Add re-admission to dataset
######################################################################################

dataset['re_admission'] = np.nan

dataset['re_admission'].loc[0] = 0
for i in dataset.index[1:]:
    if dataset['SUBJECT_ID'].loc[i] == dataset['SUBJECT_ID'].loc[i - 1]:
        dataset['re_admission'].loc[i] = 1
    else:
        dataset['re_admission'].loc[i] = 0

######################################################################################
# Add elixhauser to dataset (Use SID score)
######################################################################################

elixhauser_ICD_CODE = pd.read_csv('../data/mimiciii/elixhauser_ICD_CODE.csv')

coef_dict = {
    'aids' : 0,
    'alcohol abuse' : 0,
    'blood loss anemias' : -3,
    'cardiac arrhythmias' : 8,
    'congestive heart failure' : 9,
    'chronic pulmonary' : 3,
    'coagulopathy' : 12,
    'deficiency anemias' : 0,
    'depression' : -5,
    'diabetes complicated' : 1,
    'diabetes uncomplicated' : 0,
    'drug abuse' : -11,
    'fluid electrolyte' : 11,
    'hypertension' : -2,
    'hypothyroidism' : 0,
    'liver disease' : 7,
    'lymphoma' : 8,
    'metastatic cancer' : 17,
    'other neurological' : 5,
    'obesity' : -5,
    'paralysis' : 4,
    'peptic ucler' : 0,
    'peripheral vascular' : 4,
    'psychosis' : -6,
    'pulmonary circulation' : 5,
    'renal failure' : 7,
    'rheumatoid arthritis' : 0,
    'solid tumor' : 10,
    'valvular_disease' : 0,
    'weight_loss' : 10
}

def get_elixhauser(series, elixhauser_ICD_CODE):
    subject_id = series['SUBJECT_ID']
    hadm_id = series['HADM_ID']
    icd_code_set = set(DIAGNOSES_ICD.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}')['ICD9_CODE'].values)
    total = 0
    for key, value in coef_dict.items():
        disease_code_set = set(elixhauser_ICD_CODE[elixhauser_ICD_CODE['elixhauser instructions'] == key]['ICD9_CODE'].values)
        if len(icd_code_set.intersection(disease_code_set)) > 0:
            total += value
    return total
    
dataset['elixhauser'] = dataset.apply(get_elixhauser, axis=1, args=(elixhauser_ICD_CODE,))

######################################################################################
# Split time
######################################################################################

dataset.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
dataset['Age'] = np.nan

def split_time(hour, dataset):
    hour_period = f'0{hour}:00:00'
    index = 0
    split = pd.DataFrame(columns = dataset.columns)
    for i in tqdm(dataset.index):
        subject_id, hadm_id, dob, gender, deathtime, intime, outtime, re_admission, elixhauser = \
            dataset.loc[i, ['SUBJECT_ID', 'HADM_ID', 'DOB', 'Gender', 'DEATHTIME', 'ADMITTIME', 'DISCHTIME', 're_admission', 'elixhauser']]
        if dataset['ADMITTIME'].isnull()[i] or dataset['DISCHTIME'].isnull()[i]:
            continue
        while outtime - intime > pd.Timedelta(hour_period):
            #['SUBJECT_ID', 'HADM_ID', 'DOB', 'Gender', 'DEATHTIME', 'ADMITTIME', 'DISCHTIME', 're_admission', 'elixhauser', 'Age']
            split.loc[index] = [subject_id, hadm_id, dob, gender, deathtime, intime, 
                intime + pd.Timedelta(hour_period), re_admission, elixhauser, (intime.to_pydatetime() - dataset['DOB'].loc[i].to_pydatetime()).days / 365]
            intime += pd.Timedelta(hour_period)
            index += 1

        split.loc[index] = [subject_id, hadm_id, dob, gender, deathtime, intime, 
            outtime, re_admission, elixhauser, (intime.to_pydatetime() - dataset['DOB'].loc[i].to_pydatetime()).days / 365]
        index += 1

    split.drop('DOB', axis=1, inplace=True)
    split = split[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'Gender', 'Age', 'DEATHTIME', 're_admission', 'elixhauser']]
    split.columns = ['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME', 'Gender', 'Age', 'DEATHTIME', 're_admission', 'elixhauser']
    split.to_csv(temporal_path + f'dataset_split_{hour}_hour.csv', index=False)

processes = list()
for i in range(1, 11):
    processes.append(Process(target=split_time, args=(i, dataset)))
for i in range(10):
    processes[i].start()
for i in range(10):
    processes[i].join()