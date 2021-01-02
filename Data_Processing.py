# import all the necassary libraries
from datetime import datetime, timedelta
from dateutil import tz
import pandas as pd
import numpy as np
import random
import os
import ast

# Constants or global variable
aq_directory = 'data/aq_data/csv_file/'
mete_directory = 'data/mete_data/csv_file/station_data/'
mete_list = ['windspeed','DewPt','CloudCover','Pressure-Station','Pressure-SeaLevel','widdirec','RH','Temp']
aq_list = ['O3','SO2','NO2','PM10','PM25']
aq_check_station = {'CB_R': 'PM10', 'YL_A': 'PM25', 'MB_A': 'NO2', 'KC_A': 'SO2', 'TK_A': 'O3'}
mete_check_station = {'5': 'RH', '10': 'widdirec', '15': 'windspeed', '30': 'DewPt', '31': 'Pressure-Station',
                    '3': 'CloudCover', '8': 'Pressure-SeaLevel', '21': 'Temp'}

# Generate the mete information dataframe which contains coordinates and grid data
mete_grid_data = {'1' : [(7, 20), [113.8911, 22.3458]],
                  '10': [(21, 5), [114.0292, 22.2108]],
                  '11': [(26, 30), [114.085, 22.4339]],
                  '12': [(29, 21), [114.1067, 22.35]],
                  '13': [(29, 14), [114.1103, 22.2867]],
                  '15': [(35, 14), [114.1661, 22.2944]],
                  '16': [(35, 17), [114.1704, 22.3132]],
                  '17': [(35, 9), [114.1708, 22.2483]],
                  '18': [(36, 31), [114.1842, 22.4425]],
                  '19': [(39, 27), [114.2086, 22.4031]],
                  '2' : [(8, 11), [113.9, 22.26]],
                  '20': [(39, 16), [114.2108, 22.3111]],
                  '21': [(39, 22), [114.2153, 22.3594]],
                  '22': [(40, 6), [114.2186, 22.2142]],
                  '23': [(41, 35), [114.2351, 22.4768]],
                  '24': [(43, 17), [114.2556, 22.3158]],
                  '26': [(45, 19), [114.268, 22.3379]],
                  '29': [(45, 24), [114.2717, 22.3772]],
                  '3' : [(10, 17), [113.92, 22.32]],
                  '30': [(48, 2), [114.3006, 22.1836]],
                  '31': [(54, 34), [114.3582, 22.4729]],
                  '32': [(45, 19), [114.267, 22.3328]],
                  '33': [(35, 15), [114.17, 22.3]],
                  '34': [(35, 17), [114.17, 22.32]],
                  '4' : [(10, 16), [113.9219, 22.3094]],
                  '5' : [(15, 25), [113.9742, 22.3922]],
                  '6' : [(16, 34), [113.98, 22.47]],
                  '7' : [(16, 34), [113.9811, 22.4706]],
                  '8' : [(19, 34), [114.0089, 22.4667]],
                  '9' : [(21, 4), [114.0267, 22.2011]]
                 }
# Generate the aq information dataframe which contains coordinates and grid data
aq_grid_data = {"CB_R": [[37,13], [114.1822, 22.2819]],
                "CL_R": [[34,13],[114.1557, 22.2833]],
                "CW_A": [[33,14],[114.1429, 22.2868]],
                "EN_A": [[40,14],[114.2169, 22.2845]],
                "KC_A": [[31,22],[114.1271, 22.3586]],
                "KT_A": [[41,17],[114.2233, 22.3147]],
                "MB_A": [[55,34],[114.3583, 22.4728]],
                "MKaR": [[35,18],[114.1660, 22.3240]],
                "SP_A": [[34,19],[114.1567, 22.3315]],
                "ST_A": [[37,24],[114.1820, 22.3780]],
                "TC_A": [[13,14],[113.9411, 22.2903]],
                "TK_A": [[45,17],[114.2594, 22.3177]],
                "TM_A": [[16,25],[113.9767, 22.3908]],
                "TP_A": [[35,32],[114.1620, 22.4524]],
                "TW_A": [[30,23],[114.1121, 22.3733]],
                "YL_A": [[21,31],[114.0203, 22.4467]]
               }

# Functions
# getting all the unix timestamps within a range
def get_all_timestamps():
    start_time = 1514736000.0
    end_time = 1577804400.0
    time_r = []
    new_time = start_time
    cal_time = 0
    while cal_time != 1577808000:
        time_r.append(new_time)
        cal_time = new_time + 3600
        new_time = cal_time
    return time_r

# Merging all station data
def all_station_data(directory, pol_list, grid_data):
    time_range = get_all_timestamps()
    all_df = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"): 
            df = pd.read_csv(os.path.join(directory, filename))
            df =  pd.crosstab(index=df['time'], columns=df['type'], values=df['val'], margins=True, aggfunc='mean')
            df.reset_index(inplace = True, level = 'time')
            df = df[df.time != 'All']
            df = df.astype({'time': float})
            df = df[['time']+pol_list]
            df['loc'] = str(grid_data[filename[:-4]][1])
            lt_lg = ast.literal_eval(df['loc'].values[0])
            df_new = pd.DataFrame(columns=['time', 'lat', 'long','station_name'])
            df_new['time'], df_new['lat'], df_new['long'], df_new['station_name'] = time_range, lt_lg[1], lt_lg[0], filename[:-4]
            df_new = pd.merge(df_new, df, how='left', on='time')[['time', 'lat', 'long','station_name']+pol_list]
            all_df.append(df_new)
    all_station = pd.concat(all_df, ignore_index=False)
    print("Sanity check: ",all_station.shape, 17520 *16)
    return all_station
        
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
    df['Year'], df['Month'] ,df['Hour'], df["Week"], df["Day"] = df['HKT'].dt.year, df['HKT'].dt.month, df['HKT'].dt.hour, df['HKT'].dt.week ,df['HKT'].dt.day
    return df

# fill the missing values
def fill_nullvalues(df, list_values):
    # # Replacing Nan values in other column 
    print("Null values before preprocessing\n", df.applymap(lambda x: pd.isnull(x)).sum())
    # Replacing the Nan values with the mean groupedBy aq_id, year, month ,week, day
    df[list_values] = df.groupby(['station_name','Year','Month','Week','Day'])[list_values].transform(lambda x: x.fillna(x.mean()))
    # Replacing the remaining Nan values with the mean groupedBy aq_id, year, month ,week
    df[list_values] = df.groupby(['station_name','Year','Month','Week'])[list_values].transform(lambda x: x.fillna(x.mean()))
    # Replacing the remaining Nan values with the mean groupedBy aq_id, year, month 
    df[list_values] = df.groupby(['station_name','Year','Month'])[list_values].transform(lambda x: x.fillna(x.mean()))
    # Replacing the remaining Nan values with the mean groupedBy aq_id, year 
    df[list_values] = df.groupby(['station_name','Year'])[list_values].transform(lambda x: x.fillna(x.mean()))
    # Replacing the remaining Nan values with 0
    df = final_df.fillna(0)
    print("Null values after preprocessing\n", df.applymap(lambda x: pd.isnull(x)).sum())
    print("Shape of the final dataset", df.shape)
    return df

def final_data(name, directory, pol_list, df_grid_data):
    final_df = all_station_data(directory,pol_list, df_grid_data)
    final_df['HKT'] = final_df['time'].apply(UTC2HKT)
    final_df = ymd(final_df)
    if name == 'mete':
        final_df["widdirec"] = final_df["widdirec"].replace({990: np.nan})  
    final_df = fill_nullvalues(final_df, pol_list)
    final_df.to_csv('data/'+name+'_data/final_'+name+'_data.csv', index= False)
    return final_data

def station_info(df_grid_data, name):
    df_info = pd.DataFrame(columns=['station_name', 'lat', 'long', 'x', 'y'])
    for k,v in df_grid_data.items():
        new_row = {'station_name':k, 'lat':v[1][1], 'long':v[1][0], 'x':v[0][0], 'y':v[0][1]}
        df_info = df_info.append(new_row, ignore_index=True)
    df_info.to_csv('data/'+name +'_info.csv', index= False)
    return None

def sanity_check_1(check_station, directory, name):
    for k,v in check_station.items():
        print(k)
        df = pd.read_csv(directory+k+'.csv')
        time_list = df[(df['type']==v)].time.values.tolist()
        if len(time_list) < 10:
            secure_random = random.sample(time_list, len(time_list))
        else:
            secure_random = random.sample(time_list, 10)
        check_list = []
        for i in secure_random:
            val_1 = df[(df['time']==i) & (df['type'] == v)][['val']].values[0][0]
            final_df = pd.read_csv('data/'+name+'_data/final_'+name+'_data.csv')
            val_2 = final_df[(final_df['time']==i) & (final_df['station_name'] == k)][[v]].values[0][0]
            if val_1 == val_2:
                check_list.append('Yes')
            else:
                check_list.append('No')
        if all(x == check_list[0] for x in check_list):
            print(k+' is verfied')
    return None

def generate_npy(year,name, pol_list):
    start_time = datetime.now()
    if name =='aq':
        df_data = pd.read_csv('data/aq_data/final_aq_data.csv', usecols = ['time','station_name','PM25','O3', 'SO2','NO2','PM10','HKT','Year'])
    else:
        df_data = pd.read_csv('data/mete_data/final_mete_data.csv', usecols = ['time','station_name','windspeed','DewPt','CloudCover','Pressure-Station','Pressure-SeaLevel','widdirec','RH','Temp', 'HKT', 'Year'])
    station_info = pd.read_csv('data/'+name+'_info.csv')
    df_data = pd.merge(df_data, station_info, on='station_name')  
    year_df = df_data[(df_data['Year']== year)]
    year_df.sort_values(by=['HKT','station_name'], ascending=True, inplace = True)
    year_df.reset_index(drop = True,inplace = True)
    grouped_data = year_df.groupby(['HKT'])
    for pol in pol_list: 
        grid_data = []
        for n, group in grouped_data:
            d = np.zeros((64, 41))
            for index, row in group[['station_name',pol , 'x', 'y']].iterrows():
                d[int(row['x'])][int(row['y'])] = row[pol]
            grid_data.append(d) 
        with open('data/'+name+'_data/npy_file/'+str(year)+'/'+pol+'.npy', 'wb') as f:
            np.save(f, grid_data)
    end_time = datetime.now()
    print(end_time - start_time)
    return None

def sanity_check_2(check_station, directory, grid_data, name):
    for k,v in mete_check_station.items():
    print(k)
    df = pd.read_csv(directory+ k +'.csv')
    time_list = df[(df['type']==v)].time.values.tolist()
    if len(time_list) < 10:
        secure_random = random.sample(time_list, len(time_list))
    else:
        secure_random = random.sample(time_list, 10)
    check_list = []
    for i in secure_random:
        if UTC2HKT(i)[:4] == '2018':
            st_time = datetime.strptime(UTC2HKT('1514736000'), '%Y-%m-%d %H:%M:%S')
            data_direc = 'data/'+name+'_data/npy_file/2018'
        else:
            st_time = datetime.strptime(UTC2HKT('1546272000'), '%Y-%m-%d %H:%M:%S')
            data_direc = 'data/'+name+'_data/npy_file/2019'
        en_time = datetime.strptime(UTC2HKT(i), '%Y-%m-%d %H:%M:%S')
        diff = en_time - st_time
        days, seconds = diff.days, diff.seconds
        hours = days * 24 + seconds // 3600
        val_1 = df[(df['time']==i) & (df['type'] == v)][['val']].values[0][0]
        val_2 = np.load(os.path.join(data_direc, v +'.npy'))[hours][grid_data[k][0][0]][grid_data[k][0][1]]
        if val_1 == val_2:
            check_list.append('yes')
    if all(x == check_list[0] for x in check_list):
            print(k+' is verfied')
    return None

station_info(aq_grid_data, 'aq')
station_info(mete_grid_data, 'mete')

final_data('aq', aq_directory, aq_list, aq_grid_data)
final_data('mete', mete_directory, mete_list, mete_grid_data)

sanity_check_1(aq_check_station, aq_directory, 'aq')
sanity_check_1(mete_check_station, mete_directory, 'mete')

generate_npy(2019,'aq',aq_list)
generate_npy(2018,'aq',aq_list)
generate_npy(2019,'mete',mete_list)
generate_npy(2018,'mete',mete_list)

sanity_check_2(aq_check_station, aq_directory, aq_grid_data, 'aq')
sanity_check_2(mete_check_station, mete_directory, mete_grid_data, 'mete')