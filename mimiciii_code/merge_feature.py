######################################################################################
# Import package
######################################################################################

import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

pd.options.mode.chained_assignment = None

preprocess_path = '../data/mimiciii/preprocess_data/'
temporal_path = '../data/mimiciii/temporal_dataset/'

######################################################################################
# ## Load require data
######################################################################################

CHARTEVENTS1 = pd.read_csv(preprocess_path + 'CHARTEVENTS1.csv')
CHARTEVENTS2 = pd.read_csv(preprocess_path + 'CHARTEVENTS2.csv')
CHARTEVENTS3 = pd.read_csv(preprocess_path + 'CHARTEVENTS3.csv')
CHARTEVENTS4 = pd.read_csv(preprocess_path + 'CHARTEVENTS4.csv')
CHARTEVENTS5 = pd.read_csv(preprocess_path + 'CHARTEVENTS5.csv')
CHARTEVENTS6 = pd.read_csv(preprocess_path + 'CHARTEVENTS6.csv')
CHARTEVENTS7 = pd.read_csv(preprocess_path + 'CHARTEVENTS7.csv')
CHARTEVENTS8 = pd.read_csv(preprocess_path + 'CHARTEVENTS8.csv')
CHARTEVENTS9 = pd.read_csv(preprocess_path + 'CHARTEVENTS9.csv')
LABEVENTS = pd.read_csv(preprocess_path + 'LABEVENTS.csv')


events_to_merge = [CHARTEVENTS1, CHARTEVENTS2, CHARTEVENTS3, CHARTEVENTS4, CHARTEVENTS5, CHARTEVENTS6, CHARTEVENTS7, 
         CHARTEVENTS8, CHARTEVENTS9, LABEVENTS]

for i in tqdm(range(len(events_to_merge))):
    for column in events_to_merge[i].columns:
        if 'TIME' in column or 'DATE' in column:
            events_to_merge[i][column] = events_to_merge[i][column].apply(lambda x : pd.Timestamp(x))

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
    

for i in tqdm(range(len(events_to_merge))):
    events_to_merge[i]['VALUENUM'] = events_to_merge[i].apply(changeUnit, axis=1)

######################################################################################
# Load variable range
######################################################################################

var_range = pd.read_csv('../data/mimiciii/variable_range.csv')
var_range.index = var_range.iloc[:, 0]
var_range = var_range.iloc[:, 1:]
var_range['GROUP_ID'] = var_range['GROUP_ID'].apply(lambda x : [int(s) for s in x.split(',')] if type(x) == type('') else x)
var_range

######################################################################################
# Merge features to dataset
######################################################################################

def merge_event(dataset, feature, EVENTS, var_range):
    # find all feature records in EVENTS
    event = EVENTS[[id in var_range.loc[feature]['GROUP_ID'] for id in EVENTS['ITEMID']]]

    for i in range(len(event)):
        
        subject_id = event.iloc[i]['SUBJECT_ID']
        hadm_id = event.iloc[i]['HADM_ID']
        time = event.iloc[i]['CHARTTIME']
        
        # detect outlier value, if out of outlier range then set to missing value
        value = event.iloc[i]['VALUENUM']
        if var_range.loc[feature].notnull()[0]:
            if value > var_range.loc[feature]['OUTLIER HIGH']:
                continue
            elif value > var_range.loc[feature]['VALID HIGH']:
                value = var_range.loc[feature]['VALID HIGH']
            elif value < var_range.loc[feature]['OUTLIER LOW']:
                continue
            elif value < var_range.loc[feature]['VALID LOW']:
                value = var_range.loc[feature]['VALID LOW']
        
        index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index
        
        low, high = 0, len(index) - 1
        while low <= high:
            mid = (low + high) >> 1
            idx = index[mid]
            if time - dataset.iloc[idx]['ENDTIME'] > pd.Timedelta('0'):
                low = mid + 1
            elif time - dataset.iloc[idx]['STARTTIME'] < pd.Timedelta('0'):
                high = mid - 1
            else:
                if dataset['records'][idx] == 0:
                    dataset[feature][idx] = value
                else:
                    record = dataset['records'][idx]
                    # use mean when multiple records in same period
                    dataset[feature][idx] = value / (record + 1) + record / (record + 1) * dataset[feature][idx]
                dataset['records'][idx] += 1
                break

                    
def merge(hour, events_to_merge, var_range):
    print(f'process {hour} start running...')
    path = temporal_path + f'dataset_split_{hour}_hour.csv'

    dataset = pd.read_csv(path)

    # change time column data's type to timestamp type
    dataset['STARTTIME'] = dataset['STARTTIME'].apply(lambda x : pd.Timestamp(x))
    dataset['ENDTIME'] = dataset['ENDTIME'].apply(lambda x : pd.Timestamp(x))
    dataset['DEATHTIME'] = dataset['DEATHTIME'].apply(lambda x : pd.Timestamp(x))
    dataset = dataset.loc[:, ['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME', 'Gender', 
                              'Age', 'DEATHTIME', 're_admission']]
                   
    print(f'process {hour} finish loading dataset')

    for feature in tqdm(var_range.index):
        # count records in same period
        dataset['records'] = pd.Series([0] * len(dataset), dtype = np.int64)

        # add new feature column
        dataset[feature] = pd.Series([np.nan] * len(dataset), dtype = np.float64)

        # add data to new feature column with corresponding subject_id, hadm_id, icustay_id and time
        for event in events_to_merge:
            merge_event(dataset, feature, event, var_range)
    
    dataset.drop(['records'], axis=1, inplace=True)
    dataset.to_csv(temporal_path + f'dataset_split_{hour}_hour_after_merge.csv', index=False)
    print(f'process {hour} finish')
    

# processes = list()
# for i in range(1, 11):
#     processes.append(Process(target=merge, args=(i, events_to_merge, var_range)))

# for i in range(len(processes)):
#     processes[i].start()
# for i in range(len(processes)):
#     processes[i].join()

# assert(True)

######################################################################################
# Load dataset split hour after merge and require data
######################################################################################

OUTPUTEVENTS = pd.read_csv(preprocess_path + 'OUTPUTEVENTS.csv')
INPUTEVENTS_MV = pd.read_csv(preprocess_path + 'INPUTEVENTS_MV.csv')
INPUTEVENTS_CV = pd.read_csv(preprocess_path + 'INPUTEVENTS_CV.csv')
DIAGNOSES_ICD = pd.read_csv(preprocess_path + 'DIAGNOSES_ICD.csv')
datasets = list()
for i in tqdm(range(10)):
    datasets.append(pd.read_csv(temporal_path + f'dataset_split_{i + 1}_hour_after_merge.csv'))
    datasets[i]['STARTTIME'] = datasets[i]['STARTTIME'].apply(lambda x : pd.Timestamp(x))
    datasets[i]['ENDTIME'] = datasets[i]['ENDTIME'].apply(lambda x : pd.Timestamp(x))
    datasets[i]['DEATHTIME'] = datasets[i]['DEATHTIME'].apply(lambda x : pd.Timestamp(x))

# change time column data's type to timestamp type
OUTPUTEVENTS['CHARTTIME'] = OUTPUTEVENTS['CHARTTIME'].apply(lambda x : pd.Timestamp(x))
INPUTEVENTS_MV['STARTTIME'] = INPUTEVENTS_MV['STARTTIME'].apply(lambda x : pd.Timestamp(x))
INPUTEVENTS_MV['ENDTIME'] = INPUTEVENTS_MV['ENDTIME'].apply(lambda x : pd.Timestamp(x))
INPUTEVENTS_CV['CHARTTIME'] = INPUTEVENTS_CV['CHARTTIME'].apply(lambda x : pd.Timestamp(x))


def merge_remain_feature (dataset, period, events_to_merge, INPUTEVENTS_MV, INPUTEVENTS_CV, OUTPUTEVENTS):

    ######################################################################################
    # Output four hourly and Output total
    ######################################################################################
    def addOutputPeriod(dataset, event, period):
        print(f'process {period} start adding output hourly...')
        outputevent_id = {40055, 40056, 40057, 40069, 40085, 40094, 40096, 40405, 40428, 40473, 40651, 40715, 43175, 
                            226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 227488, 227489, 227510}
        event = event[[x in outputevent_id for x in event['ITEMID']]]
        for i in tqdm(range(len(event))):
            # add values which unit is not nan
            subject_id = event.iloc[i]['SUBJECT_ID']
            hadm_id = event.iloc[i]['HADM_ID']
            time = event.iloc[i]['CHARTTIME']
            value = event.iloc[i]['VALUE']
            if np.isnan(value) or value <= 0:
                continue
            index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index
            
            low, high = 0, len(index) - 1
            while low <= high:
                mid = (low + high) >> 1
                idx = index[mid]
                if time - dataset.iloc[idx]['ENDTIME'] > pd.Timedelta('0'):
                    low = mid + 1
                elif time - dataset.iloc[idx]['STARTTIME'] < pd.Timedelta('0'):
                    high = mid - 1
                else:
                    dataset[f'Output {period}hourly'][idx] += value
                    break
        print(f'process {period} finish adding output hourly...')

    dataset[f'Output {period}hourly'] = [0] * len(dataset)
    # addOutputPeriod(dataset, OUTPUTEVENTS, period)

    dataset['Output total'] = [0] * len(dataset)

    def addOutputTotal(dataset, period):
        print(f'process {period} adding output total...')
        dataset['Output total'][0] = dataset[f'Output {period}hourly'][0]
        for i in tqdm(range(1, 50)):
            flag = (dataset['SUBJECT_ID'][i] == dataset['SUBJECT_ID'][i - 1]) and (dataset['HADM_ID'][i] == dataset['HADM_ID'][i - 1])
            if np.isnan(dataset[f'Output {period}hourly'][i]):
                dataset['Output total'][i] = dataset['Output total'][i - 1] * flag
            else:
                dataset['Output total'][i] = dataset['Output total'][i - 1] * flag + dataset[f'Output {period}hourly'][i]
        print(f'process {period} finish adding output total...')

    # addOutputTotal(dataset, period)

    ######################################################################################
    # Urine (mL/day)
    ######################################################################################
    def addUrine(dataset, event, period):
        print(f'process {period} adding urine...')
        urine_itemid = {40055, 40056, 40057, 40069, 40085, 40086, 40094, 40096, 40405, 40428, 40473, 40651, 40715, 43175, 
                        226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 227488, 227489}
        event = event[[x in urine_itemid for x in event['ITEMID']]]

        for i in tqdm(range(len(event))):
            subject_id = event.iloc[i]['SUBJECT_ID']
            hadm_id = event.iloc[i]['HADM_ID']
            time = event.iloc[i]['CHARTTIME']
            value = event.iloc[i]['VALUE']
            if np.isnan(value) or value <= 0:
                continue
            index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index
            
            low, high = 0, len(index) - 1
            while low <= high:
                mid = (low + high) >> 1
                idx = index[mid]
                if time - dataset.iloc[idx]['ENDTIME'] > pd.Timedelta('0'):
                    low = mid + 1
                elif time - dataset.iloc[idx]['STARTTIME'] < pd.Timedelta('0'):
                    high = mid - 1
                else:
                    dataset['Urine'][idx] += value
                    break
        print(f'process {period} finish adding urine...')


    dataset['Urine'] = [0] * len(dataset)
    # addUrine(dataset, OUTPUTEVENTS, period)

    ######################################################################################
    # Input total and Input four hour
    ######################################################################################
    def addInputCVFeat (dataset, event, feat_name):
        for i in tqdm(range(len(event))):
            subject_id = event.iloc[i]['SUBJECT_ID']
            hadm_id = event.iloc[i]['HADM_ID']
            time = event.iloc[i]['CHARTTIME']
            value = event.iloc[i]['AMOUNT']
            if np.isnan(value) or value <= 0:
                continue
            index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index
            
            low, high = 0, len(index) - 1
            while low <= high:
                mid = (low + high) >> 1
                idx = index[mid]
                if time - dataset.iloc[idx]['ENDTIME'] > pd.Timedelta('0'):
                    low = mid + 1
                elif time - dataset.iloc[idx]['STARTTIME'] < pd.Timedelta('0'):
                    high = mid - 1
                else:
                    dataset[feat_name][idx] += value
                    break

                    
    def addInputMVFeat (dataset, event, feat_name):
        for i in tqdm(range(len(event))):
            subject_id = event.iloc[i]['SUBJECT_ID']
            hadm_id = event.iloc[i]['HADM_ID']
            start = event.iloc[i]['STARTTIME']
            end = event.iloc[i]['ENDTIME']
            value = event.iloc[i]['AMOUNT']
            amountuom = event.iloc[i]['AMOUNTUOM']
            rate = event.iloc[i]['RATE']
            rateuom = event.iloc[i]['RATEUOM']
            use_value = not (np.isnan(value) or value <= 0)
            use_rate = not (np.isnan(rate) or rate <= 0)
            if not use_value and not use_rate:
                continue
            index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index

            if len(index) == 0:
                print(f'SUBJECT_ID : {subject_id}, HADM_ID : {hadm_id}')
            
            low, high, idx = 0, len(index) - 1, -1
            while low <= high:
                mid = (low + high) >> 1
                idx = index[mid]
                if start - dataset.iloc[idx]['ENDTIME'] > pd.Timedelta('0'):
                    low = mid + 1
                elif start - dataset.iloc[idx]['STARTTIME'] < pd.Timedelta('0'):
                    high = mid - 1
                else:
                    break
                    
            while idx < len(dataset) and end > dataset['STARTTIME'][idx]:
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
        print(f'process {period} finish adding input_cv hourly...')
        
                    
    def addInputMVPeriod(dataset, event, period):
        print(f'process {period} adding input_mv hourly...')
        event = event[[x == 'ml' or x == 'L' or x == 'ml/hr' for x in event['AMOUNTUOM']]]
        inputevent_mv_ids = {30001, 30004, 30005, 30007, 30008, 30009, 30011, 30014, 30015, 30018, 30020, 30021, 30030, 30060, 30061, 30063, 30066, 30094, 30143, 30159, 30160, 30161, 30168, 30176, 30179, 30180, 30185,
                            30186, 30190, 30210, 30211, 30296, 30315, 30321, 30352, 30353, 30381, 40850, 41491, 42244, 42698, 42742, 45399, 46087, 46493, 46516, 220862, 220864, 220970, 220995, 225158, 225159, 225161, 
                            225168, 225170, 225171, 225823, 225825, 225827, 225828,  225941, 225943, 226089, 227531, 227533, 228341}
        event = event[[x in inputevent_mv_ids for x in event['ITEMID']]]
        addInputMVFeat (dataset, event, f'Input {period}hourly')
        print(f'process {period} finish adding input_mv hourly...')
        

    dataset[f'Input {period}hourly'] = [0] * len(dataset)
    addInputMVPeriod(dataset, INPUTEVENTS_MV, period)
    addInputCVPeriod(dataset, INPUTEVENTS_CV, period)

    def addInputTotal(dataset, period):
        print(f'process {period} adding input total...')
        dataset['Input total'][0] = dataset[f'Input {period}hourly'][0]
        for i in tqdm(range(1, 50)):
            flag = (dataset['SUBJECT_ID'][i] == dataset['SUBJECT_ID'][i - 1]) and (dataset['HADM_ID'][i] == dataset['HADM_ID'][i - 1])
            if np.isnan(dataset[f'Input {period}hourly'][i]):
                dataset['Input total'][i] = dataset['Input total'][i - 1] * flag
            else:
                dataset['Input total'][i] = dataset['Input total'][i - 1] * flag + dataset[f'Input {period}hourly'][i]
        print(f'process {period} finish adding input total...')

    dataset['Input total'] = [0] * len(dataset)
    addInputTotal(dataset, period)

    ######################################################################################
    # Dobutamine (ITEMID: 30042, 30306, 221653)
    # Dopamine (ITEMID: 30043, 30307, 221662)
    # epinephrine (ITEMID: 30044, 30119, 30309, 221289)
    # norepinephrine (ITEMID: 221906)
    ######################################################################################
    def addlast_ine(dataset, event, period, event_name, feat_name, feat_ids):
        event = event[[x in feat_ids for x in event['ITEMID']]]
        if event_name == 'M':
            addInputMVFeat(dataset, event, feat_name)
        else:
            addInputCVFeat(dataset, event, feat_name)

    dataset['Dobutamine'] = [0] * len(dataset)
    dataset['Dopamine'] = [0] * len(dataset)
    dataset['Epinephrine'] = [0] * len(dataset)
    dataset['Norepinephrine'] = [0] * len(dataset)
    addlast_ine(dataset, INPUTEVENTS_MV, 'M', 'Dobutamine', [221653])
    addlast_ine(dataset, INPUTEVENTS_CV, 'C', 'Dobutamine', [30042, 30306]) 
    addlast_ine(dataset, INPUTEVENTS_MV, 'M', 'Dopamine', [221662])
    addlast_ine(dataset, INPUTEVENTS_CV, 'C', 'Dopamine', [30043, 30307]) 
    addlast_ine(dataset, INPUTEVENTS_MV, 'M', 'Epinephrine', [221289])
    addlast_ine(dataset, INPUTEVENTS_CV, 'C', 'Epinephrine', [30044, 30119, 30309]) 
    addlast_ine(dataset, INPUTEVENTS_MV, 'M', 'Norepinephrine', [221906])

    ######################################################################################
    # max_dose_vaso
    ######################################################################################
    def addMaxDoseVaso(dataset, INPUTEVENTS_MV, INPUTEVENTS_CV):
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
        for i in tqdm(range(len(INPUTEVENTS_CV))):
            subject_id = INPUTEVENTS_CV.iloc[i]['SUBJECT_ID']
            hadm_id = INPUTEVENTS_CV.iloc[i]['HADM_ID']
            itemid = INPUTEVENTS_CV.iloc[i]['ITEMID']
            time = INPUTEVENTS_CV.iloc[i]['CHARTTIME']
            rate = INPUTEVENTS_CV.iloc[i]['RATE']
            rateuom = INPUTEVENTS_CV.iloc[i]['RATEUOM']
            value = getRateStdCV(rate, rateuom, itemid)
            if np.isnan(rate) or rate < 0 or np.isnan(value) or value < 0:
                continue
            index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index

            low, high = 0, len(index) - 1
            while low <= high:
                mid = (low + high) >> 1
                idx = index[mid]
                if time - dataset.iloc[idx]['ENDTIME'] > pd.Timedelta('0'):
                    low = mid + 1
                elif time - dataset.iloc[idx]['STARTTIME'] < pd.Timedelta('0'):
                    high = mid - 1
                else:
                    if value > dataset['max_dose_vaso'][idx]:
                        dataset['max_dose_vaso'][idx] = value
                    break
        
                    
        # inputevents_mv     
        INPUTEVENTS_MV = INPUTEVENTS_MV[[x in vasoItemId for x in INPUTEVENTS_MV['ITEMID']]]
        for i in tqdm(range(len(INPUTEVENTS_MV))):
            subject_id = INPUTEVENTS_MV.iloc[i]['SUBJECT_ID']
            hadm_id = INPUTEVENTS_MV.iloc[i]['HADM_ID']
            itemid = INPUTEVENTS_MV.iloc[i]['ITEMID']
            start = INPUTEVENTS_MV.iloc[i]['STARTTIME']
            end = INPUTEVENTS_MV.iloc[i]['ENDTIME']
            rate = INPUTEVENTS_MV.iloc[i]['RATE']
            rateuom = INPUTEVENTS_MV.iloc[i]['RATEUOM']
            value = getRateStdMV(rate, rateuom, itemid)
            if np.isnan(rate) or rate < 0 or np.isnan(value) or value < 0:
                continue
            index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index
            
            low, high, idx = 0, len(index) - 1, -1
            while low <= high:
                mid = (low + high) >> 1
                idx = index[mid]
                if start - dataset.iloc[idx]['ENDTIME'] > pd.Timedelta('0'):
                    low = mid + 1
                elif start - dataset.iloc[idx]['STARTTIME'] < pd.Timedelta('0'):
                    high = mid - 1
                else:
                    break
                    
            while idx < len(dataset) and end > dataset['STARTTIME'][idx]:
                if value > dataset['max_dose_vaso'][idx]:
                    dataset['max_dose_vaso'][idx] = value
                idx += 1
        print(f'process {period} finish adding max dose vaso...')

    dataset['max_dose_vaso'] = [0] * len(dataset)
    addMaxDoseVaso(dataset, INPUTEVENTS_MV, INPUTEVENTS_CV)

    ######################################################################################
    # Mechanical Ventilation
    ######################################################################################
    def addMechVent (dataset, events_to_merge):
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
        
        for event in events_to_merge:
            event = event[[x in MechVentItemId for x in event['ITEMD']]]
            for i in tqdm(range(len(event))):
                subject_id = event.iloc[i]['SUBJECT_ID']
                hadm_id = event.iloc[i]['HADM_ID']
                time = event.iloc[i]['CHARTTIME']
                value = event.iloc[i]['VALUE']
                itemid = event.iloc[i]['ITEMID']
                
                if itemid == 720 and value == 'Other/Remraks':
                    continue
                elif itemid == 467 and value != 'Ventilator':
                    continue
                    
                index = dataset.query(f'SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}').index

                low, high = 0, len(index) - 1
                while low <= high:
                    mid = (low + high) >> 1
                    idx = index[mid]
                    if time - dataset.iloc[idx]['ENDTIME'] > pd.Timedelta('0'):
                        low = mid + 1
                    elif time - dataset.iloc[idx]['STARTTIME'] < pd.Timedelta('0'):
                        high = mid - 1
                    else:
                        dataset['MechVent'][idx] = 1
                        break
        print(f'process {period} finish adding MechVent...')
        
    dataset['MechVent'] = [0] * len(dataset)
    addMechVent(dataset, events_to_merge)

    ######################################################################################
    # Imputation
    ######################################################################################

    def fill_forward(dataset):
        print(f'process {period} fill forward...')
        for i in tqdm(range(1, len(dataset))):
            for column in dataset.columns[8:]:
                if np.isnan(dataset[column][i]) and dataset['SUBJECT_ID'][i] == dataset['SUBJECT_ID'][i - 1] and dataset['HADM_ID'][i] == dataset['HADM_ID'][i - 1]:
                    dataset[column][i] = dataset[column][i - 1]

        print(f'process {period} finish fill forward')

    fill_forward(dataset)

    dataset.to_csv(temporal_path + f'dataset_split_{period}_hour_after_merge_all.csv', index=False)
            
processes = list()
for i in range(1, 11):
    processes.append(Process(target=merge_remain_feature, args=(datasets[i - 1], i, events_to_merge, INPUTEVENTS_MV, INPUTEVENTS_CV, OUTPUTEVENTS)))

for i in range(len(processes)):
    processes[i].start()
for i in range(len(processes)):
    processes[i].join()