# will need to cite the following website(s) later:
# https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/


# NOT COMPLETE; STILL NEED TO DETERMINE WHICH VARIABLE TO INCLUDE FOR X: DESIGN DECISION

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import texthero as hero



# load data
dataset = pd.read_csv('covid_lies.csv')



dataset.drop('misconception',axis = 1, inplace=True)
dataset.drop('tweet_id',axis = 1,inplace = True)

np_dataset = np.array(dataset)
Y = np_dataset[:,2] # label column

dataset['tweet'] = hero.clean(dataset['tweet'])
num_features = 5
dataset['tweet'] = (hero.do_tfidf(dataset['tweet'], max_features=num_features))
#expand lists into columns
tweets_df = pd.DataFrame(dataset["tweet"].to_list(), columns=['tweet_' + str(x) for x in range(num_features)])
dataset.drop('tweet',axis = 1, inplace = True)
dataset = pd.concat([dataset,tweets_df],axis = 1)
X = dataset.drop('label', axis=1)

# change na / neg / pos into numerical:
le = preprocessing.LabelEncoder()
le.fit(Y)

# transform into numerical
Y_num = le.transform(Y)

# change it back to categorical if needed
# Y = list(le.inverse_transform(Y_num))

#print(list(le.classes_))


# XGBoost -------------------------------------

# split data into train and test sets
seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y_num, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Plot Confusion Matrix

# change y_pred back to categorical name
y_pred_cat = list(le.inverse_transform(y_pred))
y_test_cat = list(le.inverse_transform(y_test))

print(y_pred_cat)

mat = confusion_matrix(y_pred_cat, y_test_cat)
names = np.unique(y_pred_cat)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()