import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
df = pd.read_csv('C:\\Users\\admin\\Downloads\\Heart_Disease_Prediction.csv')
x = df.drop(columns = "Heart Disease")
y = df['Heart Disease']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.35)
rf = RandomForestClassifier(n_estimators=500, random_state=12, max_depth=5)
rf.fit(x_train,y_train)
pickle.dump(rf, open("heartmodel.pkl", 'wb'))