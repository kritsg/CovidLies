from xgboost import XGBClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import StackingClassifier
import pandas as pd

covidlies = pd.read_excel("covid_lies.xlsx")
covidlies.drop(["misconception_id", "tweet_id"], axis=1, inplace=True)

X = covidlies.iloc[:, 0:2]
y = covidlies.iloc[:, -1]

xgb_model = XGBClassifier()
nb_model = ComplementNB()
estimator_models = [xgb_model, nb_model]

stacked_model = StackingClassifier(estimators=estimator_models)
stacked_model.fit(X, y)    # Error when trying to fit the model