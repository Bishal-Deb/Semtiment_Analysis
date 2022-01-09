import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import preprocess_kgptalkie as ps
import re
from sklearn.feature_extraction.text import TfidfVectorizer



app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vect = pickle.load(open('tfidf.pkl', 'rb'))

tfidf= TfidfVectorizer()

def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    x=request.form.get('rate')
    x=get_clean(x)
    print(x)
    prediction=model.predict(vect.transform([x]))
    output='null'
    
    if ((prediction)>=4 ):
        output='Good'
        
    elif ((prediction)==3):
        output='Normal'

    else:
        output='Bad'


    return render_template('pred.html', prediction_text='Review={}, prediction={}'.format(prediction,output))


@app.route('/laptop')
def laptop():
    return render_template('product_laptop.html')

@app.route('/mobile')
def mobile():
    return render_template('product_mobile.html')

@app.route('/tablet')
def tablet():
    return render_template('product_tablet.html')
    
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict(data)

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
