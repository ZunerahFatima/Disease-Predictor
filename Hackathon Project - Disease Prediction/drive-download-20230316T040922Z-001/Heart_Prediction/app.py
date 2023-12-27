from flask import Flask, render_template, request

import numpy as np

import pickle

app = Flask(__name__)

model = pickle.load(open("heartmodel.pkl","rb"))

@app.route('/')

def index():

    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])

def predict():

    val1 = request.form['Age']

    val2 = request.form['gender']

    val3 = request.form['chestpain']

    val4 = request.form['bloodpressure']

    val5=request.form['Cholesterol']

    val6=request.form['fbs']

    val7=request.form['ekg']

    val8=request.form['maxhr']

    val9=request.form['exercise']
    
    val10=request.form['stdepression']

    val11=request.form['slopeofst']

    val12=request.form['numberofvessels']

    val13=request.form['thallium']

    arr = np.array([val1, val2, val3, val4,val5,val6,val7,val8, val9, val10, val11, val12, val13])

    arr = arr.astype(np.float64)

    pred =model.predict([arr])

    return render_template("index.html", data = pred)


if __name__ == '__main__':

    app.run(debug=True)