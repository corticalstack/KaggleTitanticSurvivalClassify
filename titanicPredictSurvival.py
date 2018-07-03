# Jon-Paul Boyd - Kaggle - Classifier to Predict Titantic Survial 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the train dataset
dataset_train = pd.read_csv('train.csv')
dataset_train[['Embarked']] = dataset_train[['Embarked']].fillna('S')
X_train = dataset_train.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
Y_train = dataset_train.iloc[:, 1].values


# Importing the test dataset
dataset_test = pd.read_csv('test.csv')
dataset_test[['Embarked']] = dataset_test[['Embarked']].fillna('S')
X_test = dataset_test.iloc[:,  [1, 3, 4, 5, 6, 8, 10]].values


# Taking care of missing data - Train
from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_train = imputer_train.fit(X_train[:, 2:6])
X_train[:, 2:6] = imputer_train.transform(X_train[:, 2:6])

imputer_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_test = imputer_test.fit(X_test[:, 2:6])
X_test[:, 2:6] = imputer_test.transform(X_test[:, 2:6])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_sex = LabelEncoder()
X_train[:, 1] = labelencoder_sex.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder_sex.fit_transform(X_test[:, 1])

labelencoder_embark = LabelEncoder()
X_train[:, 6] = labelencoder_embark.fit_transform(X_train[:, 6])
X_test[:, 6] = labelencoder_embark.fit_transform(X_test[:, 6])

onehotencoder = OneHotEncoder(categorical_features=[0, 1, 6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
sc_test = StandardScaler()
X_train = sc_train.fit_transform(X_train)
X_test = sc_test.fit_transform(X_test)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_train, y_pred)
# Accuracy = correct divided by total

out_passengerId = dataset_test[['PassengerId']]
out_pred = pd.DataFrame(y_pred)
dataset_concat = pd.concat([out_passengerId, out_pred], axis=1)
dataset_concat.columns = ['PassengerId', 'Survived']
dataset_concat.to_csv('test_set_prediction.csv', index=False)