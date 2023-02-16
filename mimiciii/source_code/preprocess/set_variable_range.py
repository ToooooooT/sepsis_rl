import pandas as pd
import numpy as np 

index = [
         'GCS', 'SOFA', 'Albumin', 'Arterial pH', 'Calcium', 'Glucose', 
         'Hemoglobin', 'Magnesium', 'PTT', 'Postassium', 'SGPT',
         'Arterial_BE', 'BUN', 'HCO3', 'INR',
         'Arterial Lactate', 'CO2', 'Creatinine', 'Ionized Calcium',
         'PT', 'Platelets Count', 'SGOT', 'Total Bilirubin', 
         'White Blood Cell Count', 'Chloride', 'Diastolic Blood Pressure',
         'Systolic Blood Pressure', 'Mean Blood Pressure', 'PaCO2',
         'PaO2', 'FiO2', 'Respiratory rate',
         'Temperature (Celsius)', 'Weight (kg)', 'Heart Rate',
         'SpO2'
        ]

data = {
        'OUTLIER LOW': [np.nan] * len(index), 
        'VALID LOW': [np.nan] * len(index), 
        'IMPUTE': [np.nan] * len(index), 
        'VALID HIGH': [np.nan] * len(index),
        'OUTLIER HIGH': [np.nan] * len(index),
        'GROUP_ID': ['-1'] * len(index)
       }

data = pd.DataFrame(data, index=index)

# column = ('OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER_HIGH', 'GROUP_ID')
data.loc['GCS'] = [0, 0, np.nan, 15, 15, '198']
data.loc['SOFA'] = [0, 0, np.nan, 24, 24, '227428']
data.loc['Albumin'] = [0, 0.6, 3.1, 6, 60, '772, 1521, 3727, 227456, 50862'] # delete some group_id from MIMIC_EXTRACT
data.loc['Arterial pH'] = [6.7, np.nan, np.nan, np.nan, 8, '780, 1126, 3839, 4753, 223830, 50820']
data.loc['Calcium'] = [np.nan, np.nan, np.nan, np.nan, 20, '786, 1522, 3746, 225625, 50893']
data.loc['Glucose'] = [0, 33, 128, 2000, 2200, '807, 811, 1529, 3744, 220621, 225664, 226537, 50809, 50931']
data.loc['Hemoglobin'] = [0, 0, 10.2, 25, 30, '814, 220228, 50811, 51222']
data.loc['Magnesium'] = [0, 0, 2, 20, 22, '821, 1532, 220635, 50960']
data.loc['PTT'] = [0, 18.8, 34.3, 150, 150, '825, 1533, 3796, 227466, 51275']
data.loc['Postassium'] = [0, 0, 4.1, 12, 15, '829, 1535, 3792, 4194, 227442, 227464, 50822, 50971']
data.loc['SGPT'] = [np.nan, np.nan, np.nan, np.nan, 10000, '769, 3802, 50861']
data.loc['Arterial_BE'] = [-50, np.nan, np.nan, np.nan, np.nan, '74, 776, 3736, 3740, 4196, 224828, 50802']
data.loc['BUN'] = [0, 0, 23, 250, 275, '781, 1162, 225624, 51006']
data.loc['HCO3'] = [0, 0, 25, 60, 66, '227443, 50803, 50882']
data.loc['INR'] = [np.nan, np.nan, np.nan, np.nan, 20, '815, 1530, 227467, 51237']
data.loc['Arterial Lactate'] = [0, 0.4, 1.8, 30, 33, '50813']
data.loc['CO2'] = [np.nan, np.nan, np.nan, np.nan, 120, '777, 787, 3810, 50804']
data.loc['Creatinine'] = [0, 0.1, 1, 60, 66, '791, 1525, 3750, 220615, 50912, 51081']
data.loc['Ionized Calcium'] = [np.nan, np.nan, np.nan, np.nan, 5, '816, 3766, 225667, 50808']
data.loc['PT'] = [0, 9.9, 14.5, 97.1, 150, '824, 1286, 227465, 51274']
data.loc['Platelets Count'] = [0, 0, 208, 2000, 2200, '828, 3789, 227457, 51265']
data.loc['SGOT'] = [np.nan, np.nan, np.nan, np.nan, 10000, '770, 3801, 50878']
data.loc['Total Bilirubin'] = [0, 0.1, 0.9, 60, 66, '848, 1527, 1538, 225651, 225690, 50883, 50885'] # delete some group_id from MIMIC_EXTRACT
data.loc['White Blood Cell Count'] = [0, 0, 9.9, 1000, 1100, '861, 1127, 1542, 3834, 4200, 220546, 51300, 51301']
data.loc['Chloride'] = [0, 50, 104, 175, 200, '788, 1523, 3747, 4193, 220602, 226536, 50806, 50902']
data.loc['Diastolic Blood Pressure'] = [0, 0, 59, 375, 375, '8364, 8368, 8440, 8441, 8555, 220051, 220180, 224643, 225310']
data.loc['Systolic Blood Pressure'] = [0, 0, 118, 375, 375, '6, 51, 442, 6701, 220179, 224167, 225309, 227243']
data.loc['Mean Blood Pressure'] = [0, 14, 77, 330, 375, '52, 224, 443, 455, 456, 6702, 220052, 220181, 224322, 225312']
data.loc['PaCO2'] = [0, 0, 40, 200, 220, '778, 220235, 226062, 50818']
data.loc['PaO2'] = [0, 32, 112, 700, 770, '779, 50821']
data.loc['FiO2'] = [0.2, 0.21, 0.21, 1, 1.1, '189, 190, 727, 3420, 223835']
data.loc['Respiratory rate'] = [0, 0, 19, 300, 330, '614, 615, 618, 651, 3337, 3603, 220210, 224422, 224689, 224690']
data.loc['Temperature (Celsius)'] = [14.2, 26, 37, 45, 47, '676, 677, 678, 679, 3655, 223761, 223762']
data.loc['Weight (kg)'] = [0, 0, 81.8, 250, 250, '763, 224639, 226512, 226531']
data.loc['Heart Rate'] = [0, 0, 86, 350, 390, '211, 220045']
data.loc['SpO2'] = [0, 0, 98, 100, 150, '646, 834, 220277, 50817']

data.index.name = 'features'
data.to_csv('../data/mimiciii/variable_range.csv')


