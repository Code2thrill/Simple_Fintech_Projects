import pandas as pd
import numpy as np
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
from sklearn.ensemble import RandomForestClassifier
number_of_estimators = 500
RANDOM_STATE = 80
classifier = RandomForestClassifier(n_estimators=number_of_estimators,
                       random_state=RANDOM_STATE) # adjust classifier
classifier = classifier.fit(X_train_scaled, y_train.ravel()) #fitting classifier through the data
#since y is a multi dimension array, .ravel() is used to flatten the y array, need one dimension array
predictions = classifier.predict(X_test_scaled)
print(predictions)
#create a dataframe with dictionary to compare Prediction and test data result
compare = pd.DataFrame({'Prediction':predictions,'Actual':y_test.ravel()})
print(compare)

#Evaluate the model
#use sklearn accuracy_score to check the prediction result
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print(accuracy)


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


from sklearn.metrics import confusion_matrix
confusion_matrix_results = confusion_matrix(y_test,predictions)
print(confusion_matrix_results)


confusion_matrix_dataframe = pd.DataFrame(confusion_matrix_results,
                                            index=["Actual 0", "Actual 1"],
                                            columns=["Prediction 0", "Predicted 1"])
print(confusion_matrix_dataframe)