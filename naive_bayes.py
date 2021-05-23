from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#do pip install texthero==1.0.5 NOT pip install texthero 
import texthero as hero

training = pd.read_csv('covid_lies.csv')
training.drop('misconception',axis = 1, inplace=True)
training.drop('tweet_id',axis = 1,inplace = True)
Y = training['label'] # get the label here
training['tweet'] = hero.clean(training['tweet'])
num_features = 500
training['tweet'] = (hero.do_tfidf(training['tweet'], max_features=500))
#expand lists into columns
tweets_df = pd.DataFrame(training["tweet"].to_list(), columns=['tweet_' + str(x) for x in range(num_features)])
training.drop('tweet',axis = 1, inplace = True)
training = pd.concat([training,tweets_df],axis = 1)
# print(training)

training = training.drop('label', axis=1)

# print(training)

# split data into train and test sets
seed = 7
test_size = 0.3
xtrain, xtest, ytrain, ytest = train_test_split(training, Y, test_size=test_size, random_state=seed)


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