import numpy as np
import pandas as pd

def replace_categorical_by_numerical(data):
    data = data.copy()
    data.loc[:, 'Levy'] = pd.to_numeric(data['Levy'], errors='coerce').fillna(0).astype(int)
    
    data.loc[:, 'Engine volume'] = data['Engine volume'].str.replace('Turbo', '')
    data.loc[:, 'Engine volume'] = pd.to_numeric(data['Engine volume'])
    
    data.loc[:, 'Mileage'] = data['Mileage'].str.replace(' km', '')
    data.loc[:, 'Mileage'] = pd.to_numeric(data['Mileage'])
    
    return data

def column_transformations(data):
    data['Mileage_log'] = np.log(data['Mileage']).replace(-np.inf, 1e-6)
    data['Levy_log'] = np.log(data['Levy']).replace(-np.inf, 1e-6)
    data['Engine_volume_log'] = np.log(data['Engine volume']).replace(-np.inf, 1e-6)
    
    return data

def clean_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def engineering_features(df):
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Prod. year']
    
    return df

def preprocessing_pipeline(df: pd.DataFrame):
    print('Preprocessing started...')
    print(f'initial shape: {df.shape}')
    
    df = df.drop_duplicates()
    print(f'After dropping duplicates: {df.shape}')
    
    print('Replacing categorical values with numerical...')
    df = replace_categorical_by_numerical(df)
    
    df = clean_outliers(df, ['Mileage', 'Levy', 'Price', 'Engine volume'])
    print(f'After cleaning outliers: {df.shape}')
    
    print('Feature engineering...')
    df = engineering_features(df)
    
    print('Droping unnecessary columns...')
    df = df.drop(['ID', 'Doors', 'Prod. year'], axis=1)
    
    print("Final shape after preprocessing:", df.shape)
    
    return df