# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Matrix of features - Independent variables
X = dataset.iloc[:, 3:13].values
# Dependent variable vecotr - dependent var
y = dataset.iloc[:, [13]].values

# Encoding categorical data - Country (Germany, Spain, France) and Gender ( Male ,Female) to numbers ( 0,1,2...)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
# Index 1 in X[:, 1] since its country variable
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
# Index 2 in X[:, 2] since its gender variable
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Turn country variable (0 ,1 ,2) to three dummy variables , one for each country ( each varaibl contains 0 or 1)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Remove first dummy variable - now there are only 2 dummy variables
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - we need to apply feature scaling to ease all calculations (i guess it similar to normalization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Part 2 - making ANN

# Importing Keras libraries and packages
import keras
# For initializing ANN
from keras.models import Sequential
# For creating layers
from keras.layers import Dense
# For dropout regularization - dropouts apply to neurons so they can be randomly disabled at each iteration
from keras.layers import Dropout


'''
# Initializing ANN - define as a sequence of layers - create an object of sequential class, which is basically our future ANN
classifier = Sequential()

# Dense function will randomly initialize weights close to zero
# Number of rows is equal to number of independent variables we have in our matrix of features
# Sigmoid function really good for output layer - gives probability for customer leaving or staying in the bank

# Define input layer and the first hidden layer with dropout
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
# dont go over p = 0.5 ( 50% of neurons)
classifier.add(Dropout(p = 0.1))

# Define second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dropout(p = 0.1))

# Add the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting Ann to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results - get the probabilities of customers leaving the bank
y_pred = classifier.predict(X_test)

# Convert probabilities to binary result
y_pred = (y_pred > 0.5)

# Predict if one single customer is leaving
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000 ]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# K fold Cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn= build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator= classifier, X= X_train, y = y_train, cv = 10)

mean = accuracies.mean()
variance = accuracies.std()

# Dropout regularization to reduce overfitting if needed
'''
# Tuning the ANN - Grid Search - technique for finding best parameters for ANN ( batch size , number of epochs and so on)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn= build_classifier)

# Create dictionary of parameters we want to optimize
parameters = {'batch_size' : [25, 32],
              'nb_epoch' : [100, 500],
              'optimizer' : ['adam','rmsprop']}

grid_search = GridSearchCV(estimator= classifier,
                           param_grid= parameters,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_