# import library
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

#  read forecast and real data for evaluate our model 
df = pd.read_csv('C:\\Users\\sajad\\Desktop\\cap predict\\2.3\\2.3.3\\2.3.3.csv')
df2=df.dropna(subset=['Prediction','real'])
y_true = df2.real
y_pred = df2.Prediction

#  print our metrics result
MAX_ERROR = max_error(y_true, y_pred)
MSE = mean_squared_error(y_true, y_pred)
RMSE = mean_squared_error(y_true, y_pred, squared=False)
MAPE = mean_absolute_percentage_error(y_true, y_pred)
print(f'Max error ={MAX_ERROR}' ,
 f'mse = {MSE}' ,
 f'rmse = {RMSE}',
 f'mape = {MAPE}',sep='\n' )