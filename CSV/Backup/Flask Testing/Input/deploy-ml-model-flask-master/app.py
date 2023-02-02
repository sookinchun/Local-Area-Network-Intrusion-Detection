__author__ = 'kin_chun'
import numpy as np
from flask import Flask, request, jsonify, render_template, make_response , url_for,redirect
import joblib
import os
import io
import csv
import pandas


model = joblib.load("finalized_DT_model.sav")

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']

    arr = np.array([[data1, data2, data3, data4, data5, data6]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)















