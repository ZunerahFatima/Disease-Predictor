from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("anemia_model.pkl","rb"))

@app.route('/')
def index():
    return render_template("main.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1= request.form['Gender']
    val2= request.form['Hemoglobin']
    val3=request.form['MCH']
    val4= request.form['MCHC']
    val5= request.form['MCV']
    arr = np.array([val1, val2, val3,val4,val5])
    arr = arr.astype(np.float64)
    pred =model.predict([arr])

    return render_template("main.html", data=str(pred))


if __name__ == '__main__':
    app.run(debug=True)