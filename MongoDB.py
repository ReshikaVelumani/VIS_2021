from pymongo import MongoClient
import datetime 
import json
import csv
import os
import pandas as pd

client = MongoClient("127.0.0.1", 27017).HSBC_realtime_prediction
aq_collection = client.AQ_pred_history
mete_collection = client.mete_pred_history

aq_stations = {
    'CB_R': [114.1822, 22.2819],
    'CL_R': [114.1557, 22.2833],
    'MKaR': [114.1660, 22.3240],
    'CW_A': [114.1429, 22.2868],
    'EN_A': [114.2169, 22.2845],
    'KC_A': [114.1271, 22.3586],
    'KT_A': [114.2233, 22.3147],
    'ST_A': [114.1820, 22.3780],
    'SP_A': [114.1567, 22.3315],
    'TP_A': [114.1620, 22.4524],
    'MB_A': [114.3583, 22.4728],
    'TK_A': [114.2594, 22.3177],
    'TW_A': [114.1121, 22.3733],
    'TM_A': [113.9767, 22.3908],
    'TC_A': [113.9411, 22.2903],
    'YL_A': [114.0203, 22.4467]
    }
mete_stations = {"1" : [113.8911, 22.3458],
                 "2" : [113.9, 22.26 ],
                 "3" : [113.92, 22.32],
                 "4" : [113.9219, 22.3094],
                 "5" : [113.9742, 22.3922],
                 "6" : [113.98, 22.47],
                 "7" : [113.9811, 22.4706],
                 "8" : [114.0089, 22.4667],
                 "9" : [114.0267, 22.2011],
                 "10": [114.0292, 22.2108],
                 "11": [114.085, 22.4339],
                 "12": [114.1067, 22.35],
                 "13": [114.1103, 22.2867],
                 "14": [114.1536, 22.5306],
                 "15": [114.1661, 22.2944],
                 "16": [114.1704, 22.3132],
                 "17": [114.1708, 22.2483],
                 "18": [114.1842, 22.4425],
                 "19": [114.2086, 22.4031],
                 "20": [114.2108, 22.3111],
                 "21": [114.2153, 22.3594],
                 "22": [114.2186, 22.2142],
                 "23": [114.2351, 22.4768],
                 "24": [114.2556, 22.3158],
#                  "25": [114.2675, 22.3379],
                 "25": [114.268, 22.3379],
#                  "27": [114.269, 22.3369],
#                  "28": [114.269, 22.337],
                 "26": [114.2717, 22.3772],
                 "27": [114.3006, 22.1836],
                 "28": [114.3582, 22.4729],
                 "29": [114.267, 22.3328 ],
                 "30": [114.17, 22.3],
                 "31": [114.17, 22.32] }

mete_data = ['Wind', 'Temp','RH', 'CloudCover', 'DewPt', 'Pressure-SeaLevel', 'Pressure-Station']
cols = ['loc', 'time', 'type', 'val']
start_time = 1514736000
end_time = 1577804400
distance = 0.001
time_data = []
for i in range(0,17519):
    new_timestamp = start_time + 1 * 60 * 60
    start_time = new_timestamp
    time_data.append(new_timestamp)
    
for k, v in aq_stations.items():
    cursor = aq_collection.find(
        {"loc": v, 
         "time" : { "$gt": 1514732400, "$lt" : 1577804400}}, { '_id':0,'loc':1, 'time':1, 'type':1, 'val':1})
    starttime = datetime.datetime.now()
    result = []
    for doc in cursor:
        result.append(doc) 
    if not result:
        print(v)
    else:
        keys = result[0].keys()
    with open(k + '.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result)
    endtime = datetime.datetime.now()
    time_diff = endtime - starttime
    print("Finished the station {} in {}".format(k, time_diff))

for feat in mete_data:
    feature = feat
    for k, loc in mete_stations.items():
        query_list = [
                    {'time': {'$lt': 1577804400}},
                    {'time': {'$gte': 1514732400}}
                ]

        query_list.append({'type': feature})

        query_list.append({'loc': {
                        '$near': {
                            '$geometry': {
                                'type': "Point",
                                'coordinates': loc
                            },
                            '$maxDistance': distance
                        }}})

        # Get the result of the query
        results = mete_collection.find({'$and': query_list},
                                  {'_id': False, 'rid': False, 'loc': False, 'type': False})
        df = pd.DataFrame(list(results))
        if df.empty:
            print('yes')
            df = pd.DataFrame() 
            df['time'], df['val'] = time_data, None
        df['Station_Name'],  df['loc']  = k, str(loc)
        print(df.shape)
        df.to_csv('data/mete_data/csv_file/'+feature+'_data/'+k+'_'+feature+'.csv', index=False)
    print("Finished the feature: ",feature)
    
direc = 'data/mete_data/csv_file/Wind_data/'
for filename in os.listdir(direc):
    if filename.endswith(".csv"):
        print(filename)
        df_wind = pd.read_csv('data/mete_data/csv_file/Wind_data/'+filename)
        df_speed = df_wind[['time', 'val', 'Station_Name', 'loc']]
        df_direc = df_wind[['time', 'val2', 'Station_Name', 'loc']]
        df_direc.rename(columns={"val2": "val"}, inplace = True)
        df_speed.to_csv('data/mete_data/csv_file/WindSpeed_data/'+filename[:-9]+'_windspeed.csv', index=False)
        df_direc.to_csv('data/mete_data/csv_file/WindDir_data/'+filename[:-9]+'_widdirec.csv', index=False)