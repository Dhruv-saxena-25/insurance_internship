from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from flask_cors import CORS, cross_origin
from insurance.pipline.prediction_pipeline import PredictionPipeline
from insurance.constants import *
from insurance.utils.main_utils import CustomData

application=Flask(__name__)
CORS(application)
app=application

@app.route("/", methods=['GET'])
@cross_origin()



def home():
    
    return render_template('home.html')

app.route('/predict',methods=['GET', 'POST'])
def predictRoute():

    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age= request.form.get('age'),
            sex= request.form.get('sex'),
            bmi= request.form.get('bmi'),
            children= request.form.get('children'),
            smoker= request.form.get('smoker'),
            region= request.form.get('region')
        )
        dataframe = data.get_data_as_data_frame()
        print(dataframe)

        obj = PredictionPipeline()
        result = obj.predict(dataframe)
        return render_template('home.html', results=result)
    

if __name__=="__main__":
    app.run(host=APP_HOST, port= APP_PORT, debug= True)    