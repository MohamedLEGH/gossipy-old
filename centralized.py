# => test 100 times to have confident results ? or cross validation ?
# => test Logistic Regression & SVC from PyTorch ?
# => test Adaline and Perceptron from Scikit learn ?
# => test Adaline and Perceptron from scratch ?

### Spambase 
# From https://archive.ics.uci.edu/dataset/94/spambase
# Attribute Type: Real, Binary
# 4601 instances
# 57 attributes

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim

from gossipy.model.nn import AdaLine
from gossipy.model.handler import PegasosHandler, AdaLineHandler
from gossipy.data.handler import ClassificationDataHandler

df = pd.read_csv("spambase.csv")
#print(df.columns)

######## Test with all the data

logreg = LogisticRegression(max_iter=10000)
svc = LinearSVC(max_iter=10000, dual=False) # Prefer dual=False when n_samples > n_features.

X = df.loc[:, df.columns != "is_spam"]
y = df.is_spam

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=True)
print("######## Test with all the data ########\n")

# LogReg
start_time = time.time()
logreg.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = logreg.score(x_test, y_test)
print("accuracy logreg is", score)
print("time training logreg is", learning_time, "secondes")
print("number of iterations logreg is", logreg.n_iter_)

# SVC
start_time = time.time()
svc.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = svc.score(x_test, y_test)
print("accuracy svc is", score)
print("time training svc is", learning_time, "secondes")
print("number of iterations svc is", svc.n_iter_)

## test size = 0.25
# Accuracy logreg = 0.9018245004344049
# SOTA logistic regression = 0.91
# SOTA = 0.95 (Xgboost classification)

#Time training logreg 0.5178978443145752 secondes

## test size = 0.1 (same as gossip paper)
#accuracy logreg is 0.9197396963123644
#time training logreg is 0.49722909927368164 secondes
#accuracy svc is 0.911062906724512
#time training svc is 0.06670951843261719 secondes

######## Test with 1/5 of the data
print("\n######## Test with 1/5 of the data ########\n")

df_sample = df.sample(frac=0.2, random_state=42)
X = df_sample.loc[:, df_sample.columns != "is_spam"]
y = df_sample.is_spam

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=True)
logreg = LogisticRegression(max_iter=10000)
svc = LinearSVC(max_iter=10000, dual=False)

# LogReg
start_time = time.time()
logreg.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = logreg.score(x_test, y_test)
print("accuracy logreg is", score)
print("time training logreg is", learning_time, "secondes")
print("number of iterations logreg is", logreg.n_iter_)

# SVC
start_time = time.time()
svc.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = svc.score(x_test, y_test)
print("accuracy svc is", score)
print("time training svc is", learning_time, "secondes")
print("number of iterations svc is", svc.n_iter_)

## test size = 0.25
# Accuracy logreg = 0.9260869565217391
#Time training logreg 0.18087124824523926 secondes
## test size = 0.1 (same as gossip paper)
#accuracy logreg is 0.9130434782608695
#time training logreg is 0.2187948226928711 secondes
#accuracy svc is 0.8804347826086957
#time svc is 0.008883953094482422 secondes

######## Test with 1/100 of the data
print("\n######## Test with 1/100 of the data ########\n")

df_sample = df.sample(frac=0.01, random_state=42)
X = df_sample.loc[:, df_sample.columns != "is_spam"]
y = df_sample.is_spam

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2, shuffle=True)
logreg = LogisticRegression(max_iter=10000)
svc = LinearSVC(max_iter=10000, dual=False)

# LogReg
start_time = time.time()
logreg.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = logreg.score(x_test, y_test)
print("accuracy logreg is", score)
print("time training logreg is", learning_time, "secondes")
print("number of iterations logreg is", logreg.n_iter_)

# SVC
start_time = time.time()
svc.fit(x_train, y_train)
end_time = time.time()
learning_time = end_time - start_time
score = svc.score(x_test, y_test)
print("accuracy svc is", score)
print("time training svc is", learning_time, "secondes")
print("number of iterations svc", svc.n_iter_)

## test size = 0.25
# Accuracy logreg = 0.5833333333333334
#Time training logreg 0.04704427719116211 secondes
## test size = 0.1 (same as gossip learning paper)
#accuracy logreg is 0.6
#time training logreg is 0.039453744888305664 secondes
#accuracy svc is 0.8
#time training svc is 0.002503633499145508 secondes

################ Adaline and Pegasos ################
print("\n################ Adaline and Pegasos ################\n")
data = df.to_numpy()
y = LabelEncoder().fit_transform(data[:, 57])
X = np.delete(data, [57], axis=1).astype('float64')
X = StandardScaler().fit_transform(X)
X = torch.tensor(X).float()
y = torch.tensor(y).long()
y = 2*y - 1 #convert 0/1 labels to -1/1
data_handler = ClassificationDataHandler(X, y, test_size=.1)

model = AdaLine(data_handler.size(1))
learning_rate = .01

# handler = AdaLineHandler(net=model, learning_rate=learning_rate)
# #print(handler.model(data_handler.Xte))
# handler.init()
# #print(handler.model(data_handler.Xte))
# # criterion = nn.MSELoss()
# # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# num_epochs = 1000
# start_time = time.time()
# for epoch in range(num_epochs):
#     handler._update(data_handler.get_train_set())
# end_time = time.time()
# learning_time = end_time - start_time

# scores = handler.evaluate(data_handler.get_eval_set())

# print("AdaLineHandler\n")
# print(scores)
# print("time training AdaLine is", learning_time, "secondes")
# print("number of iterations AdaLine: ", num_epochs)
# No results with AdaLineHandler (NaN values or accuracy = 0)

handler = PegasosHandler(net=model, learning_rate=learning_rate)
handler.init()

num_epochs = 10
start_time = time.time()
for epoch in range(num_epochs):
    handler._update(data_handler.get_train_set())
end_time = time.time()
learning_time = end_time - start_time

scores = handler.evaluate(data_handler.get_eval_set())

# PegasosHandler :
print("PegasosHandler\n")
print(scores)
print("time training Pegasos is", learning_time, "secondes")
print("number of iterations Pegasos: ", num_epochs)

## test size : 0.1

## without label transformation
#number of iterations Pegasos:  10000
#{'accuracy': 0.3869565217391304, 'precision': 0.17198067632850242, 'recall': 0.33147113594040967, 'f1_score': 0.22646310432569972, 'auc': 0.9376727171514344}
#time training Pegasos is 128.36589860916138 secondes

## with y = 2*y - 1 : 0/1 => -1/1
#number of iterations Pegasos:  10000
#{'accuracy': 0.9282608695652174, 'precision': 0.928498985801217, 'recall': 0.9199884689556452, 'f1_score': 0.923825390532871, 'auc': 0.9678721246943279}
#time training Pegasos is 126.20047402381897 secondes

# number of iterations Pegasos:  100
# {'accuracy': 0.9260869565217391, 'precision': 0.9256561241627714, 'recall': 0.9182091095250402, 'f1_score': 0.9216040100250626, 'auc': 0.9678124813614585}
# time training Pegasos is 12.187886238098145 secondes

# number of iterations Pegasos:  10
# {'accuracy': 0.9260869565217391, 'precision': 0.9256561241627714, 'recall': 0.9182091095250402, 'f1_score': 0.9216040100250626, 'auc': 0.9680112924710231}
# time training Pegasos is 1.2688534259796143 secondes

# number of iterations Pegasos:  1
# {'accuracy': 0.9108695652173913, 'precision': 0.9130685563612393, 'recall': 0.8986560368993419, 'f1_score': 0.9047046399644285, 'auc': 0.9596612258693016}
# time training Pegasos is 0.13678455352783203 secondes

# number of iterations Pegasos:  0
# {'accuracy': 0.38913043478260867, 'precision': 0.19456521739130433, 'recall': 0.5, 'f1_score': 0.28012519561815336, 'auc': 0.5}


######## Test with 1/5 of the data
print("\n######## Test with 1/5 of the data ########\n")

df_sample = df.sample(frac=0.2, random_state=42)
data = df_sample.to_numpy()
y = LabelEncoder().fit_transform(data[:, 57])
X = np.delete(data, [57], axis=1).astype('float64')
X = StandardScaler().fit_transform(X)
X = torch.tensor(X).float()
y = torch.tensor(y).long()
y = 2*y - 1 #convert 0/1 labels to -1/1
data_handler = ClassificationDataHandler(X, y, test_size=.1)

model = AdaLine(data_handler.size(1))
learning_rate = .01
handler = PegasosHandler(net=model, learning_rate=learning_rate)
handler.init()

num_epochs = 0
start_time = time.time()
for epoch in range(num_epochs):
    handler._update(data_handler.get_train_set())
end_time = time.time()
learning_time = end_time - start_time

scores = handler.evaluate(data_handler.get_eval_set())

print("PegasosHandler\n")
print(scores)
print("time training Pegasos is", learning_time, "secondes")
print("number of iterations Pegasos: ", num_epochs)

## test size : 0.1

# number of iterations Pegasos:  0
# {'accuracy': 0.45652173913043476, 'precision': 0.22826086956521738, 'recall': 0.5, 'f1_score': 0.3134328358208955, 'auc': 0.5}

# number of iterations Pegasos:  10
# {'accuracy': 0.9565217391304348, 'precision': 0.9561904761904761, 'recall': 0.9561904761904761, 'f1_score': 0.9561904761904761, 'auc': 0.9828571428571429}
# time training Pegasos is 0.3742356300354004 secondes

# number of iterations Pegasos:  100
# {'accuracy': 0.9565217391304348, 'precision': 0.9561904761904761, 'recall': 0.9561904761904761, 'f1_score': 0.9561904761904761, 'auc': 0.9852380952380951}
# time training Pegasos is 3.377408027648926 secondes

print("\n######## Test with 1/100 of the data ########\n")

df_sample = df.sample(frac=0.01, random_state=42)
data = df_sample.to_numpy()
y = LabelEncoder().fit_transform(data[:, 57])
X = np.delete(data, [57], axis=1).astype('float64')
X = StandardScaler().fit_transform(X)
X = torch.tensor(X).float()
y = torch.tensor(y).long()
y = 2*y - 1 #convert 0/1 labels to -1/1
data_handler = ClassificationDataHandler(X, y, test_size=.2)

model = AdaLine(data_handler.size(1))
learning_rate = .01
handler = PegasosHandler(net=model, learning_rate=learning_rate)
handler.init()

num_epochs = 2
start_time = time.time()
for epoch in range(num_epochs):
    handler._update(data_handler.get_train_set())
end_time = time.time()
learning_time = end_time - start_time

scores = handler.evaluate(data_handler.get_eval_set())

print("PegasosHandler\n")
print(scores)
print("time training Pegasos is", learning_time, "secondes")
print("number of iterations Pegasos: ", num_epochs)

## test size : 0.1

# number of iterations Pegasos:  0
# {'accuracy': 0.2, 'precision': 0.1, 'recall': 0.5, 'f1_score': 0.16666666666666669, 'auc': 0.5}

# number of iterations Pegasos:  10
# {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'auc': 1.0}
# time training Pegasos is 0.01704096794128418 secondes

# number of iterations Pegasos:  100
# {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'auc': 1.0}
# time training Pegasos is 0.174515962600708 secondes

## test size : 0.2

# number of iterations Pegasos:  0
# {'accuracy': 0.3333333333333333, 'precision': 0.16666666666666666, 'recall': 0.5, 'f1_score': 0.25, 'auc': 0.5}

# number of iterations Pegasos:  1
# {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'auc': 1.0}
# time training Pegasos is 0.0030782222747802734 secondes

# number of iterations Pegasos:  2
# {'accuracy': 0.7777777777777778, 'precision': 0.875, 'recall': 0.6666666666666666, 'f1_score': 0.6785714285714286, 'auc': 1.0}
# time training Pegasos is 0.004636287689208984 secondes

# number of iterations Pegasos:  10
# {'accuracy': 0.7777777777777778, 'precision': 0.875, 'recall': 0.6666666666666666, 'f1_score': 0.6785714285714286, 'auc': 1.0}
# time training Pegasos is 0.015517473220825195 secondes

# number of iterations Pegasos:  100
# {'accuracy': 0.7777777777777778, 'precision': 0.875, 'recall': 0.6666666666666666, 'f1_score': 0.6785714285714286, 'auc': 1.0}
# time training Pegasos is 0.16272997856140137 secondes

# number of iterations Pegasos:  1000
# {'accuracy': 0.7777777777777778, 'precision': 0.875, 'recall': 0.6666666666666666, 'f1_score': 0.6785714285714286, 'auc': 1.0}
# time training Pegasos is 1.5687329769134521 secondes

## test size : 0.5

# number of iterations Pegasos:  100
# {'accuracy': 0.782608695652174, 'precision': 0.775, 'recall': 0.7619047619047619, 'f1_score': 0.7667342799188641, 'auc': 0.9126984126984127}
# time training Pegasos is 0.10532569885253906 secondes

################ Logistic Regression (Gossipy) ################
#print("\n################ Logistic Regression (Gossipy) ################\n")

#net = LogisticRegression(data_handler.Xtr.shape[1], 2)

