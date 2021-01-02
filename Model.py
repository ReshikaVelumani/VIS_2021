import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os
import datetime
from datetime import timedelta
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, BatchNormalization, concatenate, Conv2D, RepeatVector

# Parameters
N_in = 24
N_out = 1
N_x = 64
N_y = 41
N_pol = 8

# Encoder
encoder_input = Input(shape=(N_in, N_x, N_y, N_pol), name = 'encoder_input')
first_ConvLSTM = ConvLSTM2D(filters= 64, kernel_size= (3,3), padding='same', return_sequences= True, name = 'ConvLSTM_1')(encoder_input)
first_BatchNormalization = BatchNormalization()(first_ConvLSTM)
second_ConvLSTM = ConvLSTM2D(filters= 32, kernel_size= (3,3), padding='same', return_sequences= True, name = 'ConvLSTM_2')(first_BatchNormalization)
second_BatchNormalization = BatchNormalization()(second_ConvLSTM)
print(second_BatchNormalization.shape)

# Decoder
third_ConvLSTM = ConvLSTM2D(filters= 64, kernel_size= (3,3), padding='same', return_sequences= True, name = 'ConvLSTM_3')(second_BatchNormalization)
third_BatchNormalization = BatchNormalization()(third_ConvLSTM)
fourth_ConvLSTM = ConvLSTM2D(filters= 32, kernel_size= (3,3), padding='same', name = 'ConvLSTM_4')(third_BatchNormalization)
fourth_BatchNormalization = BatchNormalization()(fourth_ConvLSTM)
print(fourth_BatchNormalization.shape)
x = Model(inputs = encoder_input, outputs = fourth_BatchNormalization)

# Forecast encoder
Forecast_encoder_input = Input(shape=(N_out, N_x, N_y, 1), name = 'Forecast_encoder_input')
fifth_ConvLSTM = ConvLSTM2D(filters= 64, kernel_size= (3,3), padding='same', return_sequences= True, name = 'ConvLSTM_5')(Forecast_encoder_input)
fifth_BatchNormalization = BatchNormalization()(fifth_ConvLSTM)
sixth_ConvLSTM = ConvLSTM2D(filters= 32, kernel_size= (3,3), padding='same', name = 'ConvLSTM_6')(fifth_BatchNormalization)
sixth_BatchNormalization = BatchNormalization()(sixth_ConvLSTM)
print(sixth_BatchNormalization.shape)
y = Model(inputs = Forecast_encoder_input, outputs = sixth_BatchNormalization)

# Concatenation Layer
combined = concatenate([x.output,y.output])
final_output = Conv2D(filters=1, kernel_size=(1,1), activation='relu')(combined)
model = Model(inputs = [x.input, y.input], outputs = final_output)

# Compiling the model
model.compile(optimizer='Adam', loss='mean_squared_logarithmic_error')
model.summary()

cmaq_train = cmaq_data(train_start, train_end)
y_train = target_data(train_start, train_end)
model.fit(x = [x_train, cmaq_train], y=y_train, batch_size=16, epochs=20)
model.save('model_2.h5')

eval_start = '2018-05-02 22:00:00'
eval_end = '2018-05-04 22:00:00'
x_eval = aq_mete_data(eval_start, eval_end)
cmaq_eval = cmaq_data(eval_start, eval_end)
y_eval = target_data(eval_start, eval_end)
result = model.predict([x_eval, cmaq_data_eval])

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

def rmse(y,f):
    t = 0
    for i in range(len(y)):
        t = t+pow(y[i]-f[i],2)
    r = t/len(y)
    r = pow(r,1/2)
    return r
def IOA(obs, pred):
    return 1 -(np.sum((obs-pred)**2))/(np.sum((np.abs(pred-np.mean(obs))+np.abs(obs-np.mean(obs)))**2))