#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import numpy as np
import sys



year=int(sys.argv[2]) #2023
month=int(sys.argv[3]) #3
taxi_type=sys.argv[1]  #yellow

# !pip install scikit-learn==1.5.0

# get_ipython().system('pip freeze > requirements.txt')





# get_ipython().system('python -V')







categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    print("downloading trip data...")
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet')




def apply_model(df,model_name='model.bin'):
    with open(model_name, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    print(f"opening {model_name}...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred

std_dev_predictions = np.std(apply_model(df))
mean_dev_predictions = np.mean(apply_model(df))
print('std for pridicted duration is ',std_dev_predictions)
print('mean for pridicted duration is ',mean_dev_predictions)





df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df

df_result=pd.DataFrame()

df_result['ride_id']=df['ride_id']
df_result['predicted_duration']=apply_model(df)
print(f"model has been trained on the required dataframe...")




df_result.to_parquet(
    path=f'output/{taxi_type}/{year:04d}-{month:02d}.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)

print("saving results...")




