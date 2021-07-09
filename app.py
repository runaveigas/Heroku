# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:49:56 2021

@author: RUNA
"""

import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = np.round(prediction[0], 2)
 
    return render_template('index.html', prediction_text='The predicted Blood Pressure is {} mm Hg'.format(output))


if __name__ == "__main__":
    app.run(debug=True) 