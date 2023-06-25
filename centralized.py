# => test 100 times to have confident results ?

import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("spambase.csv")
#print(df.columns)

logreg = LogisticRegression(max_iter=10000)

X = df.loc[:, df.columns != "is_spam"]
y = df.is_spam

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)

start_time = time.time()
logreg.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = logreg.score(x_test, y_test)
print("accuracy is", score)
print("time is", learning_time, "secondes")
# Accuracy = 0.9018245004344049
# SOTA logistic regression = 0.91
# SOTA logistic regression = 0.95 (Xgboost classification)

#Time 0.5178978443145752 secondes


# Test with 1/5 of the data

df_sample = df.sample(frac=0.2, random_state=42)
X = df_sample.loc[:, df_sample.columns != "is_spam"]
y = df_sample.is_spam

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, shuffle=True)
logreg2 = LogisticRegression(max_iter=10000)

start_time = time.time()
logreg2.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = logreg2.score(x_test, y_test)
print("accuracy is", score)
print("time is", learning_time, "secondes")
# Accuracy = 0.9260869565217391
#Time 0.18087124824523926 secondes

# Test with 1/100 of the data

df_sample2 = df.sample(frac=0.01, random_state=42)
X = df_sample2.loc[:, df_sample2.columns != "is_spam"]
y = df_sample2.is_spam

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2, shuffle=True)
logreg3 = LogisticRegression(max_iter=10000)

start_time = time.time()
logreg3.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = logreg3.score(x_test, y_test)
print("accuracy is", score)
print("time is", learning_time, "secondes")
# Accuracy = 0.5833333333333334
#Time 0.04704427719116211 secondes

