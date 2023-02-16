######################################################################################
# Import package
######################################################################################

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Process

pd.options.mode.chained_assignment = None

preprocess_path = '../data/mimiciii/preprocess_data/'
temporal_path = '../data/mimiciii/temporal_dataset/'

######################################################################################
# ## Load require data
######################################################################################
print('Loading data...')

CHARTEVENTS1 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS1.csv'))
CHARTEVENTS2 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS2.csv'))
CHARTEVENTS3 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS3.csv'))
CHARTEVENTS4 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS4.csv'))
CHARTEVENTS5 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS5.csv'))
CHARTEVENTS6 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS6.csv'))
CHARTEVENTS7 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS7.csv'))
CHARTEVENTS8 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS8.csv'))
CHARTEVENTS9 = pd.read_csv(os.path.join(preprocess_path, 'CHARTEVENTS9.csv'))
LABEVENTS = pd.read_csv(os.path.join(preprocess_path, 'LABEVENTS.csv'))

chart_lab_events = [CHARTEVENTS1, CHARTEVENTS2, CHARTEVENTS3, CHARTEVENTS4, CHARTEVENTS5, CHARTEVENTS6, CHARTEVENTS7, CHARTEVENTS8, CHARTEVENTS9, LABEVENTS]

OUTPUTEVENTS = pd.read_csv(os.path.join(preprocess_path, 'OUTPUTEVENTS.csv'))
INPUTEVENTS_MV = pd.read_csv(os.path.join(preprocess_path, 'INPUTEVENTS_MV.csv'))
INPUTEVENTS_CV = pd.read_csv(os.path.join(preprocess_path, 'INPUTEVENTS_CV.csv'))

datasets = list()
for i in range(1, 11):
    datasets.append(pd.read_csv(os.path.join(temporal_path, f'dataset_split_{i}_hour_merge_lab_chart.csv')))

# change time column data's type to timestamp type
OUTPUTEVENTS['CHARTTIME'] = OUTPUTEVENTS['CHARTTIME'].apply(lambda x : pd.Timestamp(x))
INPUTEVENTS_MV['STARTTIME'] = INPUTEVENTS_MV['STARTTIME'].apply(lambda x : pd.Timestamp(x))
INPUTEVENTS_MV['ENDTIME'] = INPUTEVENTS_MV['ENDTIME'].apply(lambda x : pd.Timestamp(x))
INPUTEVENTS_CV['CHARTTIME'] = INPUTEVENTS_CV['CHARTTIME'].apply(lambda x : pd.Timestamp(x))

for i in range(len(chart_lab_events)):
    for column in chart_lab_events[i].columns:
        if 'TIME' in column or 'DATE' in column:
            chart_lab_events[i][column] = chart_lab_events[i][column].apply(lambda x : pd.Timestamp(x))

for i in range(len(datasets)):
    datasets[i]['STARTTIME'] = datasets[i]['STARTTIME'].apply(lambda x : pd.Timestamp(x))
    datasets[i]['ENDTIME'] = datasets[i]['ENDTIME'].apply(lambda x : pd.Timestamp(x))
    datasets[i]['DEATHTIME'] = datasets[i]['DEATHTIME'].apply(lambda x : pd.Timestamp(x))


var_range = pd.read_csv('../data/mimiciii/variable_range.csv')
var_range.index = var_range.iloc[:, 0]
var_range = var_range.iloc[:, 1:]
var_range['GROUP_ID'] = var_range['GROUP_ID'].apply(lambda x : [int(s) for s in x.split(',')] if type(x) == type('') else x)

print('Finish loading data')

######################################################################################
# unified unit
######################################################################################

# weight , temperature (celcius) , FiO2 (torr)

def changeUnit(series):
    # Weight lbs -> kg
    if series['ITEMID'] == 226531:
        return series['VALUENUM'] * 0.45359237
    # temperature F -> C
    elif series['ITEMID'] in [678, 679, 223761]:
        return (series['VALUENUM'] - 32) * 5 / 9
    # FiO2 decFrc -> torr
    elif series['ITEMID'] == 223835:
        return series['VALUENUM'] / 100
    return series['VALUENUM']
    

print('Unified unit...')

for i in range(len(chart_lab_events)):
    chart_lab_events[i]['VALUENUM'] = chart_lab_events[i].apply(changeUnit, axis=1)

print('Finish unified unit')

######################################################################################
# Merge features to dataset
######################################################################################

def is_time_in_state (time, start, end):
    return time - start < pd.Timedelta('0') or time - end >= pd.Timedelta('0')


def getStartIndex (index, dataset, time):
    low, high = 0, len(index) - 1
    while low <= high:
        mid = (low + high) >> 1
        idx = index[mid]
        if time - dataset.loc[idx, 'ENDTIME'] >= pd.Timedelta('0'):
            low = mid + 1
        elif time - dataset.loc[idx, 'STARTTIME'] < pd.Timedelta('0'):
            high = mid - 1
        else:
            return idx
    return index[low]


def getPatientIndex(patient_table, subject_id, hadm_id):
    hash_key = str(int(subject_id)) + '_' + str(int(hadm_id))
    return patient_table[hash_key]


def merge (dataset, period, chart_lab_events, INPUTEVENTS_MV, INPUTEVENTS_CV, OUTPUTEVENTS):

    # hash table of patient index
    patient_table = dict()
    for id in dataset.apply(lambda series: str(int(series['SUBJECT_ID'])) + '_' + str(int(series['HADM_ID'])), axis=1).unique():
        subject_id, hadm_id = id.split('_')
        patient_table[id] = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index

    def merge_chart_lab(dataset, hour, chart_lab_events, var_range):

        def merge_event(dataset, feature, EVENTS, var_range):
            # find all feature records in EVENTS
            event = EVENTS[[id in var_range.loc[feature, 'GROUP_ID'] for id in EVENTS['ITEMID']]]

            for i in event.index:
                subject_id, hadm_id, time, value =  event.loc[i, ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUENUM']]

                if value <= 0 and feature != 'Arterial_BE':
                    continue
                
                # detect outlier value, if out of outlier range then set to missing value
                if value > var_range.loc[feature, 'OUTLIER HIGH']:
                    continue
                elif value > var_range.loc[feature, 'VALID HIGH']:
                    value = var_range.loc[feature, 'VALID HIGH']
                elif value < var_range.loc[feature, 'OUTLIER LOW']:
                    continue
                elif value < var_range.loc[feature, 'VALID LOW']:
                    value = var_range.loc[feature, 'VALID LOW']
                
                index = getPatientIndex(patient_table, subject_id, hadm_id)

                # record time is not in the time when patient in the dataset
                if is_time_in_state(time, dataset.loc[index[0], 'STARTTIME'], dataset.loc[index[-1], 'ENDTIME']):
                    continue

                idx = getStartIndex(index, dataset, time)
                if dataset['records'][idx] == 0:
                    dataset[feature][idx] = value
                else:
                    record = dataset['records'][idx]
                    # use mean when multiple records in same period
                    dataset[feature][idx] = value / (record + 1) + record / (record + 1) * dataset[feature][idx]
                dataset['records'][idx] += 1

                
        print(f'process {hour} start merging chartevents and labevents...')

        for feature in tqdm(var_range.index):
            # count records in same period
            dataset['records'] = pd.Series([0] * len(dataset), dtype = np.int64)

            # add new feature column
            dataset[feature] = pd.Series([np.nan] * len(dataset), dtype = np.float64)

            # add data to new feature column with corresponding subject_id, hadm_id, icustay_id and time
            for event in chart_lab_events:
                merge_event(dataset, feature, event, var_range)
        
        dataset.drop(['records'], axis=1, inplace=True)
        print(f'process {hour} finish merging chartevents and labevents')


    merge_chart_lab(dataset, period, chart_lab_events, var_range)

    dataset.to_csv(temporal_path + f'dataset_split_{period}_hour_merge_lab_chart.csv', index=False)

    ######################################################################################
    # Output four hourly and Output total
    ######################################################################################
    def addOutputPeriod(dataset, event, period):
        print(f'process {period} start adding output hourly...')
        outputevent_id = {40055, 40056, 40057, 40069, 40085, 40094, 40096, 40405, 40428, 40473, 40651, 40715, 43175, 
                            226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 227488, 227489, 227510}
        event = event[[x in outputevent_id for x in event['ITEMID']]]
        for i in tqdm(event.index):
            subject_id, hadm_id, time, value =  event.loc[i, ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']]
            if np.isnan(value) or value <= 0:
                continue

            index = getPatientIndex(patient_table, subject_id, hadm_id)

            # record time is not in the time when patient in the dataset
            if is_time_in_state(time, dataset.loc[index[0], 'STARTTIME'], dataset.loc[index[-1], 'ENDTIME']):
                continue
            
            idx = getStartIndex(index, dataset, time)
            dataset[f'Output {period}hourly'][idx] += value

        print(f'process {period} finish adding output hourly')


    dataset[f'Output {period}hourly'] = pd.Series([0] * len(dataset), dtype = np.float64)
    addOutputPeriod(dataset, OUTPUTEVENTS, period)


    def addOutputTotal(dataset, period):
        print(f'process {period} adding output total...')
        for i in tqdm(range(1, len(dataset))):
            flag = (dataset['SUBJECT_ID'][i] == dataset['SUBJECT_ID'][i - 1]) and (dataset['HADM_ID'][i] == dataset['HADM_ID'][i - 1])
            if np.isnan(dataset[f'Output {period}hourly'][i]):
                dataset['Output total'][i] = dataset['Output total'][i - 1] * flag
            else:
                dataset['Output total'][i] = dataset['Output total'][i - 1] * flag + dataset[f'Output {period}hourly'][i]
        print(f'process {period} finish adding output total')


    dataset['Output total'] = pd.Series([0] * len(dataset), dtype = np.float64)
    addOutputTotal(dataset, period)

    ######################################################################################
    # Urine (mL/day)
    ######################################################################################
    def addUrine(dataset, event, period):
        print(f'process {period} adding urine...')
        urine_itemid = {40055, 40056, 40057, 40069, 40085, 40086, 40094, 40096, 40405, 40428, 40473, 40651, 40715, 43175, 
                        226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 227488, 227489}
        event = event[[x in urine_itemid for x in event['ITEMID']]]

        for i in tqdm(event.index):
            subject_id, hadm_id, time, value = event.loc[i, ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']]
            if np.isnan(value) or value <= 0:
                continue

            index = getPatientIndex(patient_table, subject_id, hadm_id)

            # record time is not in the time when patient in the dataset
            if is_time_in_state(time, dataset.loc[index[0], 'STARTTIME'], dataset.loc[index[-1], 'ENDTIME']):
                continue

            idx = getStartIndex(index, dataset, time)
            dataset['Urine'][idx] += value

        print(f'process {period} finish adding urine')


    dataset['Urine'] = pd.Series([0] * len(dataset), dtype = np.float64)
    addUrine(dataset, OUTPUTEVENTS, period)

    ######################################################################################
    # Input total and Input four hour
    ######################################################################################
    def addInputCVFeat (dataset, event, feat_name):
        for i in tqdm(event.index):
            subject_id, hadm_id, time, value = event.loc[i, ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'AMOUNT']]
            if np.isnan(value) or value <= 0:
                continue

            index = getPatientIndex(patient_table, subject_id, hadm_id)

            # record time is not in the time when patient in the dataset
            if is_time_in_state(time, dataset.loc[index[0], 'STARTTIME'], dataset.loc[index[-1], 'ENDTIME']):
                continue

            idx = getStartIndex(index, dataset, time)
            dataset[feat_name][idx] += value
            
                    
    def addInputMVFeat (dataset, event, feat_name):
        for i in tqdm(event.index):
            subject_id, hadm_id, start, end, value, amountuom, rate, rateuom =  \
                event.loc[i, ['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM']]

            use_value = not (np.isnan(value) or value <= 0)
            use_rate = not (np.isnan(rate) or rate <= 0)
            if not use_value and not use_rate:
                continue

            if start - end > pd.Timedelta('0'):
                continue

            index = getPatientIndex(patient_table, subject_id, hadm_id)

            # record time is not in the time when patient in the dataset
            if is_time_in_state(start, dataset.loc[index[0], 'STARTTIME'], dataset.loc[index[-1], 'ENDTIME']):
                continue

            idx = getStartIndex(index, dataset, start)

            while idx <= index[-1] and end > dataset['STARTTIME'][idx]:
                interval = min(end, dataset['ENDTIME'][idx]) - max(dataset['STARTTIME'][idx], start)
                if amountuom == 'ml/hr' and use_value:
                    interval = interval.total_seconds() / 60 / 60
                    dataset[feat_name][idx] += (value * interval)
                elif rateuom == 'mL/hour' and use_rate:
                    interval = interval.total_seconds() / 60 / 60
                    dataset[feat_name][idx] += (rate * interval)
                elif rateuom == 'mL/min' and use_rate:
                    interval = interval.total_seconds() / 60
                    dataset[feat_name][idx] += (rate * interval)
                elif end <= dataset.iloc[idx]['ENDTIME'] and use_value:
                    # end and start are in same state and no rate exist
                    dataset[feat_name][idx] += value
                elif end > dataset.iloc[idx]['ENDTIME'] and use_value:
                    # end and start are in different state and no rate exist
                    dataset[feat_name][idx] += value
                    break
                else:
                    break
                idx += 1

        
    def addInputCVPeriod(dataset, event, period):
        print(f'process {period} adding input_cv hourly...')
        event = event[[x == 'ml' or x == 'cc' for x in event['AMOUNTUOM']]]
        inputevent_cv_ids = {30001, 30004, 30005, 30007, 30008, 30009, 30011, 30014, 30015, 30018, 30020, 30021, 30030, 30060, 30061, 30063, 30066, 30094, 30143, 30159, 30160, 30161, 30168, 30176, 30179, 30180, 30185, 
                            30186, 30190, 30210, 30211, 30296, 30315, 30321, 30352, 30353, 30381, 40850, 41491, 42244, 42698, 42742, 45399, 46087, 46493, 46516, 220862, 220864, 220970, 220995, 225158, 225159, 225161, 
                            225168, 225170, 225171, 225823, 225825, 225827, 225828, 225941, 225941, 225943, 226089, 227531, 227533, 228341}
        event = event[[x in inputevent_cv_ids for x in event['ITEMID']]]
        addInputCVFeat (dataset, event, f'Input {period}hourly')
        print(f'process {period} finish adding input_cv hourly')
        
                    
    def addInputMVPeriod(dataset, event, period):
        print(f'process {period} adding input_mv hourly...')
        event = event[[x == 'ml' or x == 'L' or x == 'ml/hr' for x in event['AMOUNTUOM']]]
        inputevent_mv_ids = {30001, 30004, 30005, 30007, 30008, 30009, 30011, 30014, 30015, 30018, 30020, 30021, 30030, 30060, 30061, 30063, 30066, 30094, 30143, 30159, 30160, 30161, 30168, 30176, 30179, 30180, 30185,
                            30186, 30190, 30210, 30211, 30296, 30315, 30321, 30352, 30353, 30381, 40850, 41491, 42244, 42698, 42742, 45399, 46087, 46493, 46516, 220862, 220864, 220970, 220995, 225158, 225159, 225161, 
                            225168, 225170, 225171, 225823, 225825, 225827, 225828,  225941, 225943, 226089, 227531, 227533, 228341}
        event = event[[x in inputevent_mv_ids for x in event['ITEMID']]]
        addInputMVFeat (dataset, event, f'Input {period}hourly')
        print(f'process {period} finish adding input_mv hourly')
        

    dataset[f'Input {period}hourly'] = pd.Series([0] * len(dataset), dtype = np.float64)
    addInputMVPeriod(dataset, INPUTEVENTS_MV, period)
    addInputCVPeriod(dataset, INPUTEVENTS_CV, period)

    def addInputTotal(dataset, period):
        print(f'process {period} adding input total...')
        dataset['Input total'][0] = dataset[f'Input {period}hourly'][0]
        for i in tqdm(range(1, len(dataset))):
            flag = (dataset['SUBJECT_ID'][i] == dataset['SUBJECT_ID'][i - 1]) and (dataset['HADM_ID'][i] == dataset['HADM_ID'][i - 1])
            if np.isnan(dataset[f'Input {period}hourly'][i]):
                dataset['Input total'][i] = dataset['Input total'][i - 1] * flag
            else:
                dataset['Input total'][i] = dataset['Input total'][i - 1] * flag + dataset[f'Input {period}hourly'][i]
        print(f'process {period} finish adding input total')


    dataset['Input total'] = pd.Series([0] * len(dataset), dtype = np.float64)
    addInputTotal(dataset, period)

    ######################################################################################
    # Dobutamine (ITEMID: 30042, 30306, 221653)
    # Dopamine (ITEMID: 30043, 30307, 221662)
    # epinephrine (ITEMID: 30044, 30119, 30309, 221289)
    # norepinephrine (ITEMID: 221906)
    ######################################################################################
    def addlast_ine(period, dataset, event, event_name, feat_name, feat_ids):
        print(f'process {period} adding {feat_name} in INPUTEVENTS_{event_name}V...')
        event = event[[x in feat_ids for x in event['ITEMID']]]
        if event_name == 'M':
            addInputMVFeat(dataset, event, feat_name)
        else:
            addInputCVFeat(dataset, event, feat_name)
        print(f'process {period} finish adding {feat_name} in INPUTEVENTS_{event_name}V')


    dataset['Dobutamine'] = pd.Series([0] * len(dataset), dtype = np.float64)
    dataset['Dopamine'] = pd.Series([0] * len(dataset), dtype = np.float64)
    dataset['Epinephrine'] = pd.Series([0] * len(dataset), dtype = np.float64)
    dataset['Norepinephrine'] = pd.Series([0] * len(dataset), dtype = np.float64)
    addlast_ine(period, dataset, INPUTEVENTS_MV, 'M', 'Dobutamine', [221653])
    addlast_ine(period, dataset, INPUTEVENTS_CV, 'C', 'Dobutamine', [30042, 30306]) 
    addlast_ine(period, dataset, INPUTEVENTS_MV, 'M', 'Dopamine', [221662])
    addlast_ine(period, dataset, INPUTEVENTS_CV, 'C', 'Dopamine', [30043, 30307]) 
    addlast_ine(period, dataset, INPUTEVENTS_MV, 'M', 'Epinephrine', [221289])
    addlast_ine(period, dataset, INPUTEVENTS_CV, 'C', 'Epinephrine', [30044, 30119, 30309]) 
    addlast_ine(period, dataset, INPUTEVENTS_MV, 'M', 'Norepinephrine', [221906])

    ######################################################################################
    # max_dose_vaso
    ######################################################################################
    def addMaxDoseVaso(period, dataset, INPUTEVENTS_MV, INPUTEVENTS_CV):
        print(f'process {period} adding max dose vaso...')
        vasoItemId = [30043, 30047, 30051, 30119, 30120, 30127, 30128, 30307, 221289, 221662, 221749, 221906, 222315]
        
        def getRateStdCV(rate, rateuom, itemid):
            if itemid in [30120, 221906, 30047] and rateuom == 'mcgkgmin': # norad
                return round(rate, 3)
            elif itemid in (30120, 221906, 30047) and rateuom == 'mcgmin': # norad
                return round(rate / 80, 3)
            elif itemid in (30119, 221289) and rateuom == 'mcgkgmin': # epi
                return round(rate, 3)
            elif itemid in (30119, 221289) and rateuom == 'mcgmin': # epi
                return round(rate / 80, 3)
            elif itemid in (30051, 222315) and rate > 0.2: # vasopressin, in U/h
                return round(rate * 5 / 60, 3) 
            elif itemid in (30051, 222315) and rateuom == 'Umin' and rate < 0.2: # vasopressin
                return round(rate * 5, 3)
            elif itemid in (30051, 222315) and rateuom == 'Uhr': # vasopressin
                return round(rate * 5 / 60, 3)
            elif itemid in (30128, 221749, 30127) and rateuom == 'mcgkgmin': # phenyl
                return round(rate * 0.45, 3)
            elif itemid in (30128, 221749, 30127) and rateuom == 'mcgmin': # phenyl
                return round(rate * 0.45 / 80, 3)
            elif itemid in (221662, 30043, 30307) and rateuom == 'mcgkgmin': # dopa
                return round(rate * 0.01, 3)
            elif itemid in (221662, 30043, 30307) and rateuom == 'mcgmin': # dopa
                return round(rate * 0.01 / 80, 3)
            else:
                return np.nan
        
        
        def getRateStdMV(rate, rateuom, itemid):
            if itemid in (30120, 221906, 30047) and rateuom =='mcg/kg/min': # norad
                return round(rate, 3)  
            elif itemid in (30120, 221906, 30047) and rateuom =='mcg/min': # norad
                return round(rate / 80, 3)
            elif itemid in (30119, 221289) and rateuom == 'mcg/kg/min': # epi
                return round(rate, 3)
            elif itemid in (30119, 221289) and rateuom == 'mcg/min': # epi
                return round(rate / 80, 3)
            elif itemid in (30051, 222315) and rate > 0.2: # vasopressin, in U/h
                return round(rate * 5 / 60, 3)
            elif itemid in (30051, 222315) and rateuom == 'units/min': # vasopressin
                return round(rate * 5, 3) 
            elif itemid in (30051, 222315) and rateuom == 'units/hour': # vasopressin
                return round(rate * 5 / 60, 3)
            elif itemid in (30128, 221749, 30127) and rateuom == 'mcg/kg/min': # phenyl
                return round(rate * 0.45, 3)
            elif itemid in (30128, 221749, 30127) and rateuom == 'mcg/min': # phenyl
                return round(rate * 0.45 / 80, 3)
            elif itemid in (221662, 30043, 30307) and rateuom == 'mcg/kg/min': # dopa
                return round(rate * 0.01, 3)
            elif itemid in (221662, 30043, 30307) and rateuom == 'mcg/min':
                return round(rate * 0.01 / 80, 3)
            else:
                return np.nan
            
        
        # inputevents_cv
        INPUTEVENTS_CV = INPUTEVENTS_CV[[x in vasoItemId for x in INPUTEVENTS_CV['ITEMID']]]
        for i in tqdm(INPUTEVENTS_CV.index):
            subject_id, hadm_id, itemid, time, rate, rateuom =  INPUTEVENTS_CV.loc[i, ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'RATE', 'RATEUOM']]

            value = getRateStdCV(rate, rateuom, itemid)
            if np.isnan(rate) or rate < 0 or np.isnan(value) or value < 0:
                continue
            index = getPatientIndex(patient_table, subject_id, hadm_id)

            # record time is not in the time when patient in the dataset
            if is_time_in_state(time, dataset.loc[index[0], 'STARTTIME'], dataset.loc[index[-1], 'ENDTIME']):
                continue

            idx = getStartIndex(index, dataset, time)
            if value > dataset['max_dose_vaso'][idx]:
                dataset['max_dose_vaso'][idx] = value

        # inputevents_mv     
        INPUTEVENTS_MV = INPUTEVENTS_MV[[x in vasoItemId for x in INPUTEVENTS_MV['ITEMID']]]
        for i in tqdm(INPUTEVENTS_MV.index):
            subject_id, hadm_id, itemid, start, end, rate, rateuom =  \
                INPUTEVENTS_MV.loc[i, ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'STARTTIME', 'ENDTIME', 'RATE', 'RATEUOM']]

            value = getRateStdMV(rate, rateuom, itemid)
            if np.isnan(rate) or rate < 0 or np.isnan(value) or value < 0:
                continue

            if start - end > pd.Timedelta('0'):
                continue

            index = getPatientIndex(patient_table, subject_id, hadm_id)

            # record time is not in the time when patient in the dataset
            if is_time_in_state(start, dataset.loc[index[0], 'STARTTIME'], dataset.loc[index[-1], 'ENDTIME']):
                continue

            idx = getStartIndex(index, dataset, start)
            while idx <= index[-1] and end > dataset['STARTTIME'][idx]:
                if value > dataset['max_dose_vaso'][idx]:
                    dataset['max_dose_vaso'][idx] = value
                idx += 1

        print(f'process {period} finish adding max dose vaso')


    dataset['max_dose_vaso'] = pd.Series([0] * len(dataset), dtype = np.float64)
    addMaxDoseVaso(period, dataset, INPUTEVENTS_MV, INPUTEVENTS_CV)

    ######################################################################################
    # Mechanical Ventilation
    ######################################################################################
    def addMechVent (period, dataset, chart_lab_events):
        print(f'process {period} adding MechVent...')
        MechVentItemId = [
            720, # VentTypeRecorded
            467, # O2 delivery device == ventilator
            445, 448, 449, 450, 1340, 1486, 1600, 224687, # minute volume
            639, 654, 681, 682, 683, 684, 224685, 224684, 224686, # tidal volume
            218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747, # High/Low/Peak/Mean/Neg insp force ("RespPressure")
            221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187, # Insp pressure
            543, # PlateauPressure
            5865, 5866, 224707, 224709, 224705, 224706, # APRV pressure
            60, 437, 505, 506, 686, 220339, 224700, # PEEP
            3459, # high pressure relief
            501, 502, 503, 224702, # PCV
            223, 667, 668, 669, 670, 671, 672, # TCPCV
            157, 158, 1852, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 8382, 227809, 227810, # ETT
            224701 # PSVlevel
        ]
        
        for event in tqdm(chart_lab_events):
            event = event[[x in MechVentItemId for x in event['ITEMID']]]
            for i in event.index:
                subject_id, hadm_id, itemid, time, value = event.loc[i, ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE']]

                if itemid == 720 and value == 'Other/Remraks':
                    continue
                elif itemid == 467 and value != 'Ventilator':
                    continue
                    
                index = getPatientIndex(patient_table, subject_id, hadm_id)

                # record time is not in the time when patient in the dataset
                if is_time_in_state(time, dataset.loc[index[0], 'STARTTIME'], dataset.loc[index[-1], 'ENDTIME']):
                    continue

                idx = getStartIndex(index, dataset, time)
                dataset['MechVent'][idx] = 1

        print(f'process {period} finish adding MechVent')

        
    dataset['MechVent'] = pd.Series([0] * len(dataset), dtype = np.int64)
    addMechVent(period, dataset, chart_lab_events)

    dataset.to_csv(os.path.join(temporal_path, f'dataset_split_{period}_hour_after_merge.csv'), index=False)


processes = list()
for i in range(1, 11):
    processes.append(Process(target=merge, args=(datasets[i - 1], i, chart_lab_events, INPUTEVENTS_MV, INPUTEVENTS_CV, OUTPUTEVENTS)))

for i in range(len(processes)):
    processes[i].start()
for i in range(len(processes)):
    processes[i].join()