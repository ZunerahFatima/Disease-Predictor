from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open("diabetes.pkl","rb"))

@app.route('/')
def index():
    return render_template("main.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['Pregnancies']
    val2 = request.form['Glucose']
    val3 = request.form['Blood Pressure']
    val4 = request.form['SkinThickness']
    val5=request.form['Insulin']
    val6=request.form['BMI']
    val7=request.form['DiabetesPedigreeFunction']
    val8=request.form['Age']
    arr = np.array([val1, val2, val3, val4,val5,val6,val7,val8])
    arr = arr.astype(np.float64)
    pred =model.predict([arr])

    return render_template("main.html", data=str(pred))


if __name__ == '__main__':
    app.run(debug=True)