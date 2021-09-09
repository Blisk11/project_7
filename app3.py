from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_wtf import Form, validators  
from wtforms.fields import StringField
from wtforms import TextField, BooleanField, PasswordField, TextAreaField, validators
from wtforms.widgets import TextArea


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from toolbox.predict import *
import pandas as pd

# App config.
#DEBUG = True
app = Flask(__name__)
#app.config.from_object(__name__)
#app.config['SECRET_KEY'] = 'home_credit_default_risk'


@app.route('/credit/<id_client>', methods=['GET'])
def credit(id_client):
    
    prediction, proba = predict_flask(id_client, dataframe)

    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)

if __name__ == "__main__":
    app.run()