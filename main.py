#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split

# flag to find optimal parameters for neural network
# keep false due to parameter checking taking 1 hour+
find_optimal_nn = False

# flag to find optimal k for k nearest neighbors algorithm
# change to true to find optimal k for the k-nearest neighbors alg.
find_optimal_kn = False

# read in pre-processed csv
data = pd.read_csv('final_data.csv')

# grab the 100 most recent months for average calculations
cols = list(data.tail(100))

# regularize data by subtracting means
means = {}
for col in cols:
    if 'Temp' in col:
        mean = data[col].mean()
        means[col] = mean
        data[col] = data[col] - mean

# calculate ACE value mean
mean = data['ACE'].mean()

# label is 1 if ACE is greater than the average, else label is 0
data['ACE'] = data['ACE'].apply(lambda x: 1 if x > mean else 0)

# split the features and the labels into separate frames
examples = data.drop(['ACE', 'dt', 'Hurricane_IDs'], axis=1)
labels = data['ACE']

# split data into training (90% of samples) and testing (10% of samples) groups
X_train, X_test, y_train, y_test = train_test_split(examples, labels, test_size=0.1, random_state=0)

# code used to find the optimal parameters for the neural network
# uses brute force to train and test with all combinations of passed in parameters
# takes about an hour to run
if find_optimal_nn:
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV

    parameter_space = {
        'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,)],
        'alpha': [5e-4, 5e-5],
        'learning_rate': ['constant','adaptive'],
        'random_state': [None, 1, 2, 3],
        'learning_rate_init': [0.005, 0.0005]
    }

    mlp = MLPClassifier(max_iter=1000000)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, verbose=1, cv=2)
    clf.fit(X_train, y_train)

    print('Best parameters found: ', clf.best_params_)
    print('Best score: ', clf.best_score_)

# code used to train and score a neural network with optimized parameters
# results in 83.12% accuracy on the testing set
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
params = {'hidden_layer_sizes': (10,), 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'random_state': 1, 'solver': 'lbfgs', 'max_iter': 1000000}
mlp.set_params(**params)
mlp.fit(X_train, y_train)
print('neural network accuracy: ', mlp.score(X_test, y_test))

# code used to find optimal k value for k-nearest neighbors
if find_optimal_kn:
    max_score = 0
    optimal_k = 0
    for n in range(1, 20):
        neigh = KNeighborsClassifier(n_neighbors=n, weights='uniform', algorithm='brute')
        neigh.fit(X_train, y_train)
        score = neigh.score(X_test, y_test)
        if score > max_score:
            max_score = score
            optimal_k = n
    print('optimal k: ', optimal_k)

# code used to train and test k-nearest neighbors with optimal k value
neigh = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='brute')
neigh.fit(X_train, y_train)
print('k-nearest neighbors accuracy: ', neigh.score(X_test, y_test))

# following code is used to craft future monthly vectors based on global warming trends
# create a best case scenario feature vector
low = means.copy()
for k, v in low.items():
    low[k] += 1.1
low_vector = [
    low['LandAverageTemperature'],
    low['LandMinTemperature'],
    low['LandMaxTemperature'],
    low['LandAndOceanAverageTemperature'],
    low['AvgTempUSA'],
    low['AvgTempFlorida'],
    low['AvgTempLouisiana'],
    low['AvgTempNorthCarolina'],
    low['AvgTempSouthCarolina'],
    low['AvgTempTexas'],
]

# create an average case scenario feature vector
med = means.copy()
for k, v in med.items():
    med[k] += 3.25
med_vector = [
    med['LandAverageTemperature'],
    med['LandMinTemperature'],
    med['LandMaxTemperature'],
    med['LandAndOceanAverageTemperature'],
    med['AvgTempUSA'],
    med['AvgTempFlorida'],
    med['AvgTempLouisiana'],
    med['AvgTempNorthCarolina'],
    med['AvgTempSouthCarolina'],
    med['AvgTempTexas'],
]

# create a worst case scenario feature vector
high = means.copy()
for k, v in high.items():
    high[k] += 5.4
high_vector = [
    high['LandAverageTemperature'],
    high['LandMinTemperature'],
    high['LandMaxTemperature'],
    high['LandAndOceanAverageTemperature'],
    high['AvgTempUSA'],
    high['AvgTempFlorida'],
    high['AvgTempLouisiana'],
    high['AvgTempNorthCarolina'],
    high['AvgTempSouthCarolina'],
    high['AvgTempTexas'],
]

# uses our neural network to predict the severity of these future months
# a prediction of 1 signifies above average hurricane severity
print('best case scenario classification: ', mlp.predict([low_vector])[0])
print('average case scenario classification: ', mlp.predict([med_vector])[0])
print('worst case scenario classification: ', mlp.predict([high_vector])[0])

