import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz

# Formula for the timestamp conversion(UTC to HKT)
def UTC2HKT(timestamp):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('Asia/Hong_Kong')
    fmt = '%Y-%m-%d %H:%M:%S'

    HK_T = datetime.strptime(str(datetime.utcfromtimestamp(float(timestamp))),
                            '%Y-%m-%d %H:%M:%S').replace(tzinfo=from_zone)\
                            .astimezone(to_zone).strftime(fmt)
    return HK_T

# Getting the year, month, hour, week, day from the timestamp
def ymd(df):
    df['HKT'] = pd.to_datetime(df['HKT'])
    df['Year'], df['Month'] = df['HKT'].dt.year, df['HKT'].dt.month
    return df

def date_coord_datafarme():
    df_x = pd.DataFrame(columns = ['x'], data = np.arange(0,64,1))
    df_y = pd.DataFrame(columns = ['y'], data = np.arange(0,41,1))
    df_xy = pd.merge(df_x.assign(key=0), df_y.assign(key=0), on='key').drop('key', axis=1)

    date_2018 = np.load('data/2018_timedata.npy')
    date_2019 = np.load('data/2019_timedata.npy')
    date_list = pd.DataFrame(columns=['Datetime'], data = np.concatenate((date_2018, date_2019), axis=0))
    date_list['HKT'] = date_list['Datetime'].apply(UTC2HKT)
    df_date_xy = pd.merge(df_xy.assign(key=0), date_list.assign(key=0), on='key').drop('key', axis=1)
    df_date_xy = ymd(df_date_xy)
    df_date_xy.sort_values(by=['HKT','x'], inplace= True)
    df_date_xy.reset_index(drop=True, inplace = True)
    print("Sanity check: {} and {}".format(df_date_xy.shape, 64*41*17520))
    return df_date_xy

def using_multiindex(A, columns, pollutant):
    shape = A.shape
    index = pd.MultiIndex.from_product([range(s)for s in shape], names=columns)
    df = pd.DataFrame({pollutant: A.flatten()}, index=index).reset_index()
    return df

start_time = datetime.now()
pol_list = ['O3', 'PM25', 'NO2', 'SO2', 'PM10']
df_concat = []
for pol in pol_list:
    pol_2018 = np.load('data/aq_data/npy_file/2018/'+ pol +'.npy')
    pol_2019 = np.load('data/aq_data/npy_file/2019/'+ pol +'.npy')
    pol_data = np.concatenate((pol_2018, pol_2019), axis=0)
    df = using_multiindex(pol_data, ['z','x', 'y'] ,pol)
    df_concat.append(df)
    
dfs = [df.set_index(['z', 'x', 'y']) for df in df_concat]
data_df = pd.concat(dfs, axis=1).reset_index()
data_df['key'] = data_df.index
data_df.drop(['z'], axis =1, inplace = True)

datetime_df = date_coord_datafarme()
datetime_df['key'] = datetime_df.index
result = pd.merge(data_df, datetime_df[['HKT', 'key', 'Year', 'Month']], on='key', how='left')
result.sort_values(by=['x','y'], inplace= True)
result.reset_index(drop = True, inplace= True)
print(result.head())
end_time = datetime.now() 

result_1 = result.melt(id_vars=["x", "y", "key", "HKT",'Year', 'Month'], 
        var_name="Pollutant", 
        value_name="0")

df_concat = []
for pol in ['NO2', 'O3', 'SO2', 'PM25', 'PM10']:
    df_n = result_1[result_1['Pollutant']== pol]
    lags = range(1,12) 
    res = df_n.assign(**{
        '{}'.format(t): df_n['0'].shift(t)
        for t in lags
    })
    res.dropna(inplace = True)
    df_concat.append(res)
final = pd.concat(df_concat)

eval_data = final[(final['Year'] == 2019) & (final['Month'] == 4) | (final['Month'] == 8) | (final['Month'] == 12)]
train_data_1 = final[final['Year'] == 2018]
train_data_2 = final[(final['Year'] == 2019)& ((final['Month'] == 1) | (final['Month'] == 2) | 
                                               (final['Month'] == 3) | (final['Month'] == 5) | 
                                               (final['Month'] == 6) | (final['Month'] == 7) | 
                                               (final['Month'] == 9) | (final['Month'] == 10)| 
                                               (final['Month'] == 11))]
train_data = pd.concat([train_data_1, train_data_2])

start_time = datetime.now()
grouped_df = eval_data.groupby(['HKT'])
eval_data_list = []
for name, df in grouped_df:
    df.drop(['0', 'key', 'HKT', 'Year', 'Month'], axis=1, inplace = True)
    filtered_df = df.set_index(['x', 'y', 'Pollutant'])
    new_data = filtered_df.to_xarray().to_array()
    eval_data_list.append(new_data.values)
end_time = datetime.now()   
a = np.array(eval_data_list)
np.save('eval_data.npy', a)
print(end_time - start_time)

start_time = datetime.now()
grouped_df = train_data.groupby(['HKT'])
train_data_list = []
for name, df in grouped_df:
    df.drop(['0', 'key', 'HKT', 'Year', 'Month'], axis=1, inplace = True)
    filtered_df = df.set_index(['x', 'y', 'Pollutant'])
    new_data = filtered_df.to_xarray().to_array()
    train_data_list.append(new_data.values)
end_time = datetime.now()   
a = np.array(train_data_list)
np.save('train_data.npy', a)
print(end_time - start_time)

eval_data.to_csv('eval_df.csv',index=False )
train_data.to_csv('eval_df.csv',index=False )