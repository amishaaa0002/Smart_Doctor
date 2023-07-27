from flask import Flask, render_template,request
import pandas as pd
import numpy as np

import os,sys

import pickle
import sklearn
    # used for create dataframe i.e into table from csv file
from sklearn.model_selection import train_test_split  # for splitting of the data into tranning and testing data
from sklearn.linear_model import LogisticRegression #ml algo
from sklearn.metrics import accuracy_score # it is used to check how well our model is performing

heart_path = 'C:\\Users\\reema\\Documents\\machine learning projects\heart_data.pkl'
heart_model_path = 'C:\\Users\\reema\\Documents\\machine learning projects\model.pkl'

cancer_model_path = 'C:\\Users\\reema\\Documents\\machine learning projects\model_cancer.pkl'


model = pickle.load(open(heart_model_path,'rb'))
model_cancer = pickle.load(open(cancer_model_path,'rb'))

heart_features =['age','sex',
'chest pain',
'resting blood pressure','serum cholestoral in mg/dl',
'fasting blood sugar > 120 mg/dl','resting electrocardiographic results (values 0,1,2)','maximum heart rate achieved',
'exercise induced angina','oldpeak = ST depression induced by exercise relative to rest',
'the slope of the peak exercise ST segment',
'number of major vessels (0-3) colored by flourosopy',
'thal']

cancer_features=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean'
,'compactness_mean','concavity_mean','concave points_mean','symmetry_mean',	'fractal_dimension_mean'
,'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
'concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
'perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst',
'concave points_worst','symmetry_worst','fractal_dimension_worst']


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cancer')
def breastRender():
    return render_template('bcancer.html', cancer_features=list(cancer_features))

@app.route('/heart')
def heartRender():
    return render_template('heart.html',
                           heart_features=list(heart_features))

@app.route('/predict_heart',methods=['post'])
def predict_heart():

    int_features = [int(x) for x in request.form.values()]
    features = np.asarray(int_features).reshape(1,-1)

    prediction = model.predict(features)

    return render_template('heart_yes.html', prediction_text=int(prediction))



@app.route('/predict_cancer',methods=['post'])
def predict_cancer():

    int_cancer_features = [float(x) for x in request.form.values()]
    features_cancer = np.asarray(int_cancer_features).reshape(1, -1)

    prediction_cancer = model_cancer.predict(features_cancer)

    return render_template('cancer_yes.html', prediction__cancer_text=int(prediction_cancer))



if __name__ == '__main__':
    app.run(debug=True)
