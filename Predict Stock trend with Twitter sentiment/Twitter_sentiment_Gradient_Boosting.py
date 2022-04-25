import pandas as pd
import numpy as np
from pandas.core.common import random_state
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

#import csv files as pandas dataframe
stock_prices_dataframe = pd.read_csv('stock_prices.csv')
twitter_sentiment_dataframe = pd.read_csv('twitter_sentiment.csv')
# print(stock_prices_dataframe.head())
# print(twitter_sentiment_dataframe.head())

#data processing
twitter_sentiment_dataframe = twitter_sentiment_dataframe.dropna(axis=1)#axis=1 means checking rows
#merge two dataframes on a common column "Date"
merged_dataframe = stock_prices_dataframe.merge(twitter_sentiment_dataframe,on="Date")

#notice this is an inner join
#processing and sorting sentiment data
df = merged_dataframe[["Date",
                       "Adj Close",
                       "Volume",
                       "ts_polarity",
                       "twitter_volume"]]
df.set_index("Date",inplace=True)

#sort ts_polarity in groups
Positive_sentiment_threshold = 0.11
Negative_sentiment_threshold = 0.06
sentiments = []
for sentiment_score in df['ts_polarity']:
    if sentiment_score >= Positive_sentiment_threshold:
        sentiments.append('Positive')
    elif sentiment_score <= Negative_sentiment_threshold:
        sentiments.append('Negative')
    else:
        sentiments.append('Neutral')
df['Sentiment']=sentiments
# print(df['Sentiment'].value_counts())
df['Price Difference'] = df['Adj Close'].diff()#calculate price diffrence
df.dropna(inplace=True)
RISE = 1
FALL = 0
df['Stock Trend'] = np.where(df['Price Difference']>0, RISE, FALL)
# Binary encoding for sentiments
new_df = df[['Adj Close','Volume','twitter_volume','Sentiment','Stock Trend']]
new_df = pd.get_dummies(new_df,columns=['Sentiment'])
#split and scale data
X = new_df.copy() #X is the independent Variable
X.drop('Stock Trend',axis='columns', inplace=True) #'Stock Trend' is the target result
y = new_df['Stock Trend'].values.reshape(-1,1) #y is the dependent Variable
#The new shape should be compatible with the original shape;
# numpy allow us to give one of new shape parameter as -1; t simply means that it is an unknown dimension and we want numpy to figure it out.
# y is a numpy array
#split into training data set and testing data set; 80% training and 20% testing
SPLIT = int(0.8*len(X))
X_train = X[:SPLIT]
X_test = X[SPLIT:]
y_train = y[:SPLIT]
y_test = y[SPLIT:]
# data scaling Since volume data and Adj Close data are not in the same scale
scaler = StandardScaler() #pick StandardScaler
X_scaler = scaler.fit(X_train) #fit scaler to the data
X_train_scaled = X_scaler.transform(X_train) # use data scaler scales the data
X_test_scaled = X_scaler.transform(X_test)


from sklearn.ensemble import GradientBoostingClassifier
learning_rates=[0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
NUMBER_OF_ESTIMATORS = 30
MAX_NUMBER_OF_FEATURES = 2
MAX_DEPTH = 3
RANDOM_STATE_CONSTANT = 0
for learning_rate in learning_rates:
    classifier =GradientBoostingClassifier(n_estimators=NUMBER_OF_ESTIMATORS,
                                           learning_rate=learning_rate,
                                           max_features=MAX_NUMBER_OF_FEATURES,
                                           max_depth=MAX_DEPTH,
                                           random_state=RANDOM_STATE_CONSTANT)
    classifier.fit(X_train_scaled,y_train.ravel())
    training_score = classifier.score(X_train_scaled, y_train.ravel())
    testing_score = classifier.score(X_test_scaled, y_test.ravel())
    print('Learning rate: ', learning_rate)
    print('Training Accuracy: ', training_score)
    print('Testing Accuracy: ', testing_score)
    print()
#notice as training score increases, the testing_score decreases;
#it means better training score with lower testing_score might result overfitting
#pick learning_rate: 1.5 with best Training Accuracy
LEARNING_RATE = 1.5
classifier=GradientBoostingClassifier(n_estimators=NUMBER_OF_ESTIMATORS,
                                      learning_rate=LEARNING_RATE,
                                      max_features=MAX_NUMBER_OF_FEATURES,
                                      max_depth=MAX_DEPTH,
                                      random_state=RANDOM_STATE_CONSTANT)
classifier.fit(X_train, y_train.ravel())
training_score=classifier.score(X_train_scaled, y_train)
testing_score = classifier.score(X_test_scaled, y_test)
print('Training score: ', training_score)
print('Testing score: ', testing_score)
predictions = classifier.predict(X_test_scaled)
print(pd.DataFrame({
    "Predictions": predictions,
    "Actual Stock Trend": y_test.ravel()
}))
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print(accuracy)
#Evaluate the model result
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix_results = confusion_matrix(y_test,predictions)
confusion_matrix_dataframe = pd.DataFrame(confusion_matrix_results,
                                           index = ['Actual 0', 'Actual 1'],
                                           columns = ['Predicted 0', 'Predicated 1'])
print(confusion_matrix_dataframe)
