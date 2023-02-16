######################################################################################
# Import Package
######################################################################################

import pandas as pd
import os
import numpy as np
from multiprocessing import Process

pd.options.mode.chained_assignment = None

######################################################################################
# Load Data
######################################################################################
temporal_path = '../../data/temporal_dataset/'

datasets = list()
for i in range(1, 11):
    datasets.append(pd.read_csv(os.path.join(temporal_path, f'dataset_split_{i}_hour_after_merge.csv')))


var_range = pd.read_csv('../../data/variable_range.csv')
var_range.index = var_range.iloc[:, 0]
var_range = var_range.iloc[:, 1:]
var_range['GROUP_ID'] = var_range['GROUP_ID'].apply(lambda x : [int(s) for s in x.split(',')] if type(x) == type('') else x)

def main_execute(period, dataset, var_range):
    ######################################################################################
    # Imputation
    ######################################################################################
    def imputation(period, dataset, var_range):
        print(f'process {period} start imputation...')

        idxs = var_range.index
        # fill foward
        dataset[idxs].fillna(method='ffill', inplace=True)

        # fill mean
        for idx in idxs:
            if np.isnan(var_range.loc[idx, 'IMPUTE']):
                mean_value = dataset[idx].mean()
            else:
                mean_value = var_range.loc[idx, 'IMPUTE']
            dataset[idx].fillna(value=mean_value, inplace=True)

        print(f'process {period} finish imputation')


    imputation(period, dataset, var_range)

    ######################################################################################
    # Merge shock_index
    ######################################################################################
    def get_shock_index(series):
        try:
            return series['Heart Rate'] / series['Systolic Blood Pressure']
        except:
            return np.nan


    dataset['Shock Index'] = dataset.apply(get_shock_index, axis=1)

    ######################################################################################
    # SIRS Criteria (>= 2 meets SIRS definition)
    # 1. Temp > 38C (100.4F) or < 36C (96.8F)
    # 2. Heart rate > 90
    # 3. Respiratory rate > 20 or PaCO2 < 32 mmHg
    # 4. WBC > 12,000/mm^3, < 4,000/$mm^3, or > 10% bands (?) (mm^3 = 1/1000 K/ul)
    ######################################################################################
    def get_SIRS(series):
        sum = 0
        if series['Temperature (Celsius)'] > 38 or series['Temperature (Celsius)'] < 36:
            sum += 1
        if series['Heart Rate'] > 90:
            sum += 1
        if series['Respiratory rate'] > 20 or series['PaCO2'] < 32:
            sum += 1
        if series['White Blood Cell Count'] > 12 or series['White Blood Cell Count'] < 4:
            sum += 1
        return sum

        
    dataset['SIRS'] = dataset.apply(get_SIRS, axis=1)

    ######################################################################################
    # PaO2/FiO2 ratio
    ######################################################################################
    def get_PaO2_FiO2_ratio(series):
        try:
            return series['PaO2'] / series['FiO2']
        except:
            return np.nan


    dataset['PaO2_FiO2_ratio'] = dataset.apply(get_PaO2_FiO2_ratio, axis=1)

    ######################################################################################
    # SOFA

    # PaO2/FiO2 mmHg                 
    # Platelets x 10^3 /μL    
    # Glasgow Coma Scale             
    # Bilirubin, mg/dL (*17.1 = μmol/L)   
    # Mean arterial pressure OR administration of vasoactive agents required   
    # Creatinine, mg/dL (μmol/L) (or urine output) 
    ######################################################################################
    def get_SOFA(series):
        score = 0
        # PaO2/FiO2 mmHg
        if series['PaO2_FiO2_ratio'] < 100 and series['MechVent']:
            score += 4
        elif series['PaO2_FiO2_ratio'] < 200 and series['MechVent']:
            score += 3
        elif series['PaO2_FiO2_ratio'] < 300:
            score += 2
        elif series['PaO2_FiO2_ratio'] < 400:
            score += 1
        
        # Platelets, x10^3/μL
        if series['Platelets Count'] < 20:
            score += 4
        elif series['Platelets Count'] < 50:
            score += 3
        elif series['Platelets Count'] < 100:
            score += 2
        elif series['Platelets Count'] < 150:
            score += 1
            
        # GCS (Glasgow Coma Scale)
        if series['GCS'] < 6:
            score += 4
        elif series['GCS'] < 10:
            score += 3
        elif series['GCS'] < 13:
            score += 2
        elif series['GCS'] < 15:
            score += 1
        
        # bilirubin, mg/dL (μmol/L)
        if series['Total Bilirubin'] >= 12: # > 204
            score += 4
        elif series['Total Bilirubin'] >= 6: # >= 102
            score += 3
        elif series['Total Bilirubin'] >= 2: # >= 33
            score += 2
        elif series['Total Bilirubin'] >= 1.2: # >= 20
            score += 1

        # Mean arterial pressure OR administration of vasoactive agents required
        # get Dopamine and epinephrine
        if series['Dopamine'] > 15 or series['Epinephrine'] > 0.1 or series['Norepinephrine']:
            score += 4
        elif series['Dopamine'] > 5 or series['Epinephrine'] <= 0.1 or series['Norepinephrine'] <= 0.1:
            score += 3
        elif series['Dopamine'] > 0 or series['Dobutamine'] > 0:
            score += 2
        elif series['Mean Blood Pressure'] < 70: 
            score += 1

        
        # Creatinine, mg/dL (μmol/L) (or urine (mL/day) output)
        if series['Creatinine'] >= 5.0 or series['Urine'] < 200:
            score += 4
        elif series['Creatinine'] >= 3.5 or series['Urine'] < 500:
            score += 3
        elif series['Creatinine'] >= 2.0:
            score += 2
        elif series['Creatinine'] >= 1.2:
            score += 1
        
        return score

        
    dataset['SOFA'] = dataset.apply(get_SOFA, axis=1)

    ######################################################################################
    # cumulative balance
    ######################################################################################
    def cumulatedBalanceTev(series):
        return series['Input total'] - series['Output total']


    dataset['cumulated_balance_tev'] = dataset.apply(cumulatedBalanceTev, axis=1)

    ######################################################################################
    # export unnormalization data
    ######################################################################################
    dataset.columns = ['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME', 'gender', 'age',
       'death', 're_admission', 'elixhauser', 'GCS', 'SOFA', 'Albumin',
       'Arterial_pH', 'Calcium', 'Glucose', 'Hemoglobin', 'Magnesium', 'PTT',
       'Potassium', 'SGPT', 'Arterial_BE', 'BUN', 'HCO3', 'INR',
       'Arterial_lactate', 'CO2', 'Creatinine', 'Ionized_Ca', 'PT',
       'Platelets_count', 'SGOT', 'Total_bili', 'WBC_count',
       'Chloride', 'DiaBP', 'SysBP', 'MeanBP', 'PaCO2', 'PaO2', 'FiO2', 'RR',
       'Temp_C', 'Weight_kg', 'HR', 'SpO2',
       f'output_{period}hourly', 'output_total', 'Urine', f'input_{period}hourly',
       'input_total', 'Dobutamine', 'Dopamine', 'Epinephrine',
       'Norepinephrine', 'max_dose_vaso', 'mechvent', 'shock_index', 'SIRS', 'PaO2_FiO2_ratio', 'cumulative_balance']

    dataset.to_csv(os.path.join(temporal_path, f'dataset_split_{period}_hour_unnorm.csv'), index=False)

    ######################################################################################
    # Normalization
    ######################################################################################
    def normalization(period, dataset):
        binary_features = ['gender', 're_admission', 'mechvent']
        norm_features = ['age', 'elixhauser', 'GCS', 'SOFA', 'Albumin', 'Arterial_pH', 
                        'Calcium', 'Glucose', 'Hemoglobin', 'Magnesium', 'PTT', 'Potassium', 'Arterial_BE',
                        'HCO3', 'Arterial_lactate', 'CO2', 'Ionized_Ca', 'PT', 'Platelets_count', 'WBC_count', 'Chloride', 
                        'DiaBP', 'SysBP', 'MeanBP', 'PaCO2', 'PaO2', 'FiO2', 'RR', 'Temp_C', 
                        'Weight_kg', 'HR', 'shock_index', 'SIRS', 'PaO2_FiO2_ratio', 'cumulative_balance']
        log_norm_features = ['SGPT', 'BUN', 'INR', 'Creatinine', 'SGOT', 'Total_bili', 'SpO2', 'input_total',
                        f'input_{period}hourly', 'output_total', f'output_{period}hourly']

        print(f'process {period} start normalization...')

        dataset[binary_features] = dataset[binary_features] - 0.5
        
        for feature in norm_features:
            avg = dataset[feature].mean()
            std = dataset[feature].std()
            dataset[feature] = (dataset[feature] - avg) / std 

        dataset[log_norm_features] = np.log(0.1 + dataset[log_norm_features])
        for feature in log_norm_features:
            avg = dataset[feature].mean()
            std = dataset[feature].std()
            dataset[feature] = (dataset[feature] - avg) / std

        print(f'process {period} finish normalization')

    
    normalization(period, dataset)
    dataset['death'] = dataset['death'].fillna('').apply(lambda x: 1 if len(x) > 0 else 0)
    dataset.drop(['STARTTIME', 'ENDTIME'], axis=1, inplace=True)
    dataset.to_csv(os.path.join(temporal_path, f'dataset_{period}.csv'), index=False)


processes = list()
for i in range(10):
    processes.append(Process(target=main_execute, args=(i + 1, datasets[i], var_range)))

for i in range(len(processes)):
    processes[i].start()
for i in range(len(processes)):
    processes[i].join()