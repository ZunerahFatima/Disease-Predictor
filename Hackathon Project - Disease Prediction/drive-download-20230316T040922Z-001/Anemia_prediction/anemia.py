import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
df=pd.read_csv('C:\\Users\\admin\\Downloads\\anemia.csv')
df=df.replace({'Result':{0:'Not Anemic', 1:'Anemic'}})
df
y=df["Result"]
x=df.drop('Result',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
logreg=LogisticRegression(solver='liblinear')
logreg.fit(x_train,y_train)
pickle.dump(logreg, open('anemia_model.pkl', 'wb'))

