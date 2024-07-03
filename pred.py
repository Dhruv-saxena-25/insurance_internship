
from insurance.pipline.prediction_pipeline import PredictionPipeline
import pandas as pd



# data = pd.DataFrame([[20, 'female', 25, 2, 'yes', 'southwest']], columns= ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
# print(data)

data = pd.DataFrame([[25, 'male', 21.5, 1, 'no', 'southeast']], columns= ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
print(data)

pipe = PredictionPipeline()

expense = pipe.predicts(data)
print(expense)
