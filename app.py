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
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'home_credit_default_risk'

class SimpleForm(Form):
    form_id = TextField('id:', validators=[validators.required()])
    
    @app.route("/", methods=['GET', 'POST'])
    def form():
        form = SimpleForm(request.form)
        print(form.errors)

        if request.method == 'POST':
            form_id=request.form['id']
            print(form_id)
            return(redirect('credit/'+form_id)) 
    
        if form.validate():
            # Save the comment here.
            flash('You have requested customer ID : ' + form_id)
            redirect('')
        else:
            flash('Please enter a user ID ')
    
        return render_template('formulaire_id.html', form=form)


@app.route('/credit/<id_client>', methods=['GET'])
def credit(id_client):

    #récupération id client depuis argument url
    #id_client = request.args.get('id_client', default=1, type=int)
    
    #DEBUG
    #print('id_client : ', id_client)
    #print('shape df ', dataframe.shape)
    
    #calcul prédiction défaut et probabilité de défaut
    prediction, proba = predict_flask(id_client, dataframe)

    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)

if __name__ == "__main__":
    app.run(debug=True)

