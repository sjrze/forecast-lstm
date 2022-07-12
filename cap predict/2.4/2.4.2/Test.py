# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

# read data 
df = pd.read_csv('2.4\\2.4.2\\2.4.2.csv',usecols=[0,1])
# df = df.iloc[:2].values
# df = df.reset_index()


# data describtion
df.describe()
df = df.set_index("time")
df.plot()
# plt.savefig('cap4.jpg', format = 'jpg', dpi = 600)

# make all data as train file
train = df

# normalization
scaler = MinMaxScaler()
scaler.fit(train)
scaler.fit(train)
train = scaler.transform(train)

# make generator for forecast with dynamic input based on our work
n_input = 32
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=1)

# define our model
model=Sequential()
model.add(LSTM(150,return_sequences=True,activation='relu', dropout= 0.05,input_shape=(train.shape[0],n_features)))
model.add(LSTM(100,return_sequences=True,activation='tanh',dropout= 0.04))
model.add(LSTM(100,activation='tanh',))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))


# model fiting 
a = model.fit_generator(generator,epochs=100)

plt.plot(a.history['loss'])

# our pediction numeric data
pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    pred_list.append(model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

# make forecast for our data with dynamic rate
add_dates = [df.index[-1] + int(x) for x in range(0,n_input+1)]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

#  delete dublicate data
df = df[~df.index.duplicated(keep='first')]
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])

df_proj = pd.concat([df,df_predict], axis=1)


# plot our result
plt.figure(figsize=(20, 7))
plt.plot(df_proj.index, df_proj['Normalized Capacitance Value including Faults'])
plt.plot(df_proj.index, df_proj['Prediction'], color='r', label = 'predict')
plt.legend()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
# plt.savefig('2.jpeg', format = 'png', dpi = 600)


