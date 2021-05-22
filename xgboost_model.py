# will need to cite the following website(s) later:
# https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/


# NOT COMPLETE; STILL NEED TO DETERMINE WHICH VARIABLE TO INCLUDE FOR X: DESIGN DECISION

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = pd.read_csv('covid_lies.csv')
np_dataset = np.array(dataset)

# split X and Y
X = np_dataset[:,:4] # ARE WE USING ALL OF THEM? ------------------
Y = np_dataset[:,4] # label column


# THE FOLLOWING CODES WILL NOT WORK -----------------------

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)