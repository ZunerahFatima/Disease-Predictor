import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('C:\\Users\\admin\\Downloads\\diabetes.csv')

columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = df[columns]
df=df.replace({'Outcome':{0:'No', 1:'Yes'}})
df


X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)
filename = 'Diabetes_model.pkl'
pickle.dump(classifier, open("diabetes.pkl", 'wb'))