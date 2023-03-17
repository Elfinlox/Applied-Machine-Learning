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

# Defining input values to test the score function on
ham = "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight"
spam = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005"
threshold = 0.5

# Defining Unit Tests

# Smoke Test: Function returns values without crashing
def test_smoke(text=ham, threshold=threshold, model=spam_detectorNB) -> None:
    label, prop = score(text, model, threshold)

    assert label != None, "Prediction is not boolean"
    assert prop != None, "Propensity is not float"

# Format Test: Check function input/output types 
def test_input_formats(text=ham, threshold=threshold, model=spam_detectorNB) -> None:
    label, prop = score(text, model, threshold)

    assert type(text) == str
    assert type(threshold) == float 
    assert type(label) == bool
    assert type(prop) == np.float64 

# Prediction Value Test
def test_pred_value(text=ham, threshold=threshold, model=spam_detectorNB) -> None:
    label, prop = score(text, model, threshold)
    print("Testing Prediction Value")
    assert label == False or label == True

# Propensity Value Test
def test_prop_value(text=ham, threshold=threshold, model=spam_detectorNB) -> None:
    label, prop = score(text, model, threshold)

    assert prop >= 0 and prop <= 1

# Check Prediction Value for Threshold = 0 Test
def test_pred_thres_zero(text=ham, threshold=threshold, model=spam_detectorNB) -> None:
    label, prop = score(text, model, threshold=0)

    assert label == True

# Check Prediction Value for Threshold = 1 Test
def test_pred_thres_one(text=ham, threshold=threshold, model=spam_detectorNB) -> None:
    label, prop = score(text, model, threshold=1)

    assert label == False

# Check Prediction Value for Threshold = 1 Test
def test_obvious_spam(text=spam, threshold=threshold, model=spam_detectorNB) -> None:
    label, prop = score(text, model, threshold)

    assert label == True, "Prob: {}".format(prop)

# Check Prediction Value for Threshold = 1 Test
def test_obvious_ham(text=ham, threshold=threshold, model=spam_detectorNB) -> None:
    label, prop = score(text, model, threshold)

    assert label == False

def test_flask():
        # Launch the Flask app using os.system
        os.system('python app.py')

        # Make a request
        response = requests.get('http://127.0.0.1:5000/')
        print(response.status_code)

        assert response.status_code == 200
        assert type(response.text) == str

        # Shut down the Flask app
        os.kill(os.getpid(), signal.SIGINT)