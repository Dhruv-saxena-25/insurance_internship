
from insurance.pipline.prediction_pipeline import PredictionPipeline

import numpy as np
import pandas as pd 
import pickle
import streamlit as st

st.set_page_config(layout="wide")


def show_predict_page():
	
	st.markdown(f'''<h1 style="color:white;font-size:35px;
	 text-align:center;">{"Welcome To Insurance Premium Predator APP"}</h1>''', unsafe_allow_html=True)

	# Creating form field
	with st.form('form',clear_on_submit=False):
		age = st.text_input('Age',placeholder='Age')
		sex = st.selectbox("Sex",('Male','Female'), placeholder= 'Sex')
		bmi = st.text_input('Bmi',placeholder='Bmi')
		children = st.text_input('Children', placeholder='Number of Children')
		smoker = st.selectbox('Smoker',('Yes','No'), placeholder="Choose an option",)
		reg = ('Northeast','Northwest','Southeast','Southwest')
		region = st.selectbox('Region',reg, placeholder= 'Region')

		st.markdown(""" <style> div.stButton > button:first-child {background-color:green;
			width:600px;color:white; margin: 0 auto; display: block;} </style>""", unsafe_allow_html=True)

		predict = st.form_submit_button("Predict Premium")

		if predict:

			X = pd.DataFrame([[int(age), sex, float(bmi), int(children), smoker, region]], 
	                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
			print(X)        
            
			model = PredictionPipeline()
			premium = model.predicts(X)

			st.subheader(f'Insurance Premium: ${premium[0]:.2f}')


print(show_predict_page())