from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#do pip install texthero==1.0.5 NOT pip install texthero 
import texthero as hero

training = pd.read_csv('covid_lies.csv')
training.drop('misconception',axis = 1, inplace=True)
training.drop('tweet_id',axis = 1,inplace = True)
training['tweet'] = hero.clean(training['tweet'])
training['tweet'] = (hero.do_tfidf(training['tweet'], max_features=30))
#expand lists into columns
tweets_df = pd.DataFrame(training["tweet"].to_list(), columns=['tweet_' + str(x) for x in range(30)])
training.drop('tweet',axis = 1, inplace = True)
training = pd.concat([training,tweets_df],axis = 1)
print(training)
test = training.copy()
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
plt.show()