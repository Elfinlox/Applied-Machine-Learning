import pickle
import numpy as np

from flask import Flask, request, render_template, url_for, redirect
from score import *

app = Flask(__name__, template_folder = './template')

# Importing Model

nb_path = "./models/nb_model.sav"
lr_path = "./models/lr_model.sav"
rf_path = "./models/rf_model.sav"

spam_detectorNB = pickle.load(open(nb_path, "rb"))
spam_detectorLR = pickle.load(open(lr_path, "rb"))
spam_detectorRF = pickle.load(open(rf_path, "rb"))

# Setting threshold value
threshold=0.5

@app.route('/') 
def home():
    return render_template('spam.html')

@app.route('/spam', methods=['POST'])
def spam():
    text = request.form['text']
    label, propensity = score(text, spam_detectorNB, threshold)
    propensity = round(propensity, 3)
    label = "Spam" if label == 1 else "Not spam"
    text1 = f"""The input text: {text}"""
    label1 = f"""The prediction: {label}""" 
    propensity1 = f"""The propensity score: {propensity}"""
    return render_template('result.html', text = text1, label = label1, propensity = propensity1)

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0')