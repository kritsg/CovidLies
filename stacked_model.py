from xgboost import XGBClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import StackingClassifier
import pandas as pd
import texthero as hero

training = pd.read_csv("covid_lies.csv")
training.drop(["misconception_id", "tweet_id"], axis=1, inplace=True)

training['tweet'] = hero.clean(training['tweet'])
num_features = 500
training['tweet'] = (hero.do_tfidf(training['tweet'], max_features=500))

tweets_df = pd.DataFrame(training["tweet"].to_list(), columns=['tweet_' + str(x) for x in range(num_features)])
training.drop('tweet',axis = 1, inplace = True)
training = pd.concat([training,tweets_df],axis = 1)

X = training.drop(["label"], axis=1)
y = training[["label"]]

xgb_model = XGBClassifier()
nb_model = ComplementNB()
estimator_models = [xgb_model, nb_model]

stacked_model = StackingClassifier(estimators=estimator_models)
stacked_model.fit(X, y)    # Error when trying to fit the model