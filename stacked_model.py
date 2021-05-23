# https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e

from xgboost import XGBClassifier
from sklearn.naive_bayes import ComplementNB
from vecstack import stacking
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import texthero as hero

training = pd.read_csv("covid_lies.csv")
training.drop(["misconception", "tweet_id"], axis=1, inplace=True)

training['tweet'] = hero.clean(training['tweet'])
num_features = 500
training['tweet'] = (hero.do_tfidf(training['tweet'], max_features=500))

tweets_df = pd.DataFrame(training["tweet"].to_list(), columns=['tweet_' + str(x) for x in range(num_features)])
training.drop('tweet',axis = 1, inplace = True)
training = pd.concat([training,tweets_df],axis = 1)

# Encode labels into numeric values
le = LabelEncoder()
training["label"] = le.fit_transform(training["label"])

X = training.drop(["label"], axis=1)
y = training[["label"]]
y = y.to_numpy().flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgb_model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric="mlogloss")
nb_model = ComplementNB()
estimator_models = [xgb_model, nb_model]

stack_train, stack_test = stacking(estimator_models, X_train, y_train, X_test, regression=False, mode="oof_pred_bag", needs_proba=False, save_dir=None, metric=accuracy_score, n_folds=4, stratified=True, shuffle=True, random_state=0, verbose=2)

model = xgb_model
model = model.fit(stack_train, y_train)
y_pred = model.predict(stack_test)

print(f"Final prediction score: {accuracy_score(y_test, y_pred):.5f}")