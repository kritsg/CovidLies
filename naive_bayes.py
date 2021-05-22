from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from vectorize import vectorize_col

training = pd.read_csv('covid_lies.csv')
# tweets_vectorized = 
vectorize_col(training['tweet'])
test = pd.read_csv('covid_lies.csv')
training.drop('tweet_id',axis = 1)
test.drop('tweet_id',axis = 1)
#Training/Test sets
xtrain = training.drop('label', axis=1)
ytrain = training.loc[:, 'label']
xtest = test.drop('label', axis=1)
ytest = test.loc[:, 'label']

model = ComplementNB()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)

#Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')