import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import pickle
df = pd.read_csv('C:\\Users\\admin\\Downloads\\bmd.csv')
df = df.replace({'M':1,'F':0,'No medication':0,'Glucocorticoids':1,'Anticonvulsant':2,'no fracture':0,'fracture':1})
df.drop(df.columns[[0,7]],axis = 1,inplace=True)
cols_to_scale = ['age','weight_kg','height_cm']
#create and fit scaler
scaler = StandardScaler()
scaler.fit(df[cols_to_scale])
#scale selected data
df[cols_to_scale] = scaler.transform(df[cols_to_scale])
df = pd.get_dummies(df, columns=['sex', 'fracture','medication'], drop_first=True)
x = df.drop(columns = 'bmd')
y = df['bmd']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
pickle.dump(dt, open("bmdmodel.pkl", 'wb'))