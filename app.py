import pickle as pkl
from flask import Flask, jsonify, request, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# load the model and transformations
lr_model = pkl.load(open('lr.pkl', 'rb'))
std = pkl.load(open('std.pkl', 'rb'))
log = pkl.load(open('log.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    arr_data = np.array(list(data.values()))
    print(arr_data)
    to_std = std.transform(arr_data[0:2].reshape(1,-1))
    to_log = log.transform(arr_data[2:4].reshape(1,-1))
    final_data = np.concatenate((to_std, to_log), axis=1)
    print(f'final data {final_data}')
    output = lr_model.predict(final_data)
    return jsonify(str(output[0]))

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    arr_data = np.array(data)
    to_std = std.transform(arr_data[0:2].reshape(1,-1))
    to_log = log.transform(arr_data[2:4].reshape(1,-1))
    final_data = np.concatenate((to_std, to_log), axis=1)
    output = lr_model.predict(final_data)
    result_string = "original" if output == 1 else "Fake"
    return render_template("home.html", prediction_text=f'The bank note is {result_string}')

if __name__=="__main__":
    app.run(debug=True)