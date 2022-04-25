import pandas as pd
import numpy as np

df = pd.read_csv('frequently_asked_questions.csv')
print(df)
df.dropna(inplace=True)
# vectorize data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((df.Question, df.Answer)))
#notice df.Question but not df['Question'] because it is a numpy function
#it turns pandas dataframe columns into a numpy array;
#vectorizer.fit() function only takes numpy array but not pandas dataframe
vectorized_questions = vectorizer.transform(df.Question)
print(vectorized_questions)
from sklearn.metrics.pairwise import cosine_similarity
while True:
    user_input = input("Please enter your question below: \n")
    print(user_input)
    vectorized_user_input = vectorizer.transform([user_input])
    #put input string into an array and transform it into vectors
    #find similarities between input string and questions
    similarities = cosine_similarity(vectorized_user_input, vectorized_questions)
    closest_question = np.argmax(similarities, axis=1)#axis=1 because it's multi-dimensional array
    print('Similarities: ',similarities)
    print('Closest Question: ',closest_question)
    answer = df.Answer.iloc[closest_question].values[0]
    print('Answer: ',answer)
    break

