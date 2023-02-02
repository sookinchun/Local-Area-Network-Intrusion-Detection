__author__ = 'kin_chun'
import numpy as np
from flask import Flask, request, jsonify, render_template, make_response , url_for,redirect
import joblib
import os
import io
import csv
import pandas


#create flask append
app = Flask(__name__, template_folder='templates') 

#load joblib model
model = joblib.load("finalized_DT_model.sav")

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploader', methods=["POST"])
def upload_file():
    f = request.files['file']

    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    #set the read line to first line
    stream.seek(0)
    result = transform(stream.read())

    response = make_response(result)
    #(redownload the file)
    response.headers["Content-Disposition"] = "attachment; filename=result.csv" 
    #redirect("http://localhost:5000/kinwah", code=302)
    #redirect(url_for('foo'))
    return response
    
    dataset = pandas.read_csv("C:/Users/Admin/Downloads/result.csv")
    print (dataset)
    


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


