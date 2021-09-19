
from flask import Flask, render_template
#from toolbox import predict
from random import sample

app = Flask(__name__)

@app.route("/home")
@app.route("/",methods = ["GET"])
def homepage():
    return render_template('index.html' )


"""
@app.route("/", methods = ["GET", 'POST'])
def hello():
    if request.method == "POST":
        customer_id = request.form['customer_id']
        #print(sample(test_df.index.to_list(),3))

        initial_proba = predict.predict_single_proba(customer_id)
              
        formated_probability = 'Customer default score: ' + "{0:.2f}%".format(initial_proba*100)

    return render_template('index.html', cust_probability = formated_probability)
"""
"""
@app.route("/sub", methods = ['POST'])
def submit():
    if request.method == "POST":
        name = request.form["username"]
    return render_template("sub.html", n = name)
"""


if __name__ == "__main__":
    app.run(debug = True)

