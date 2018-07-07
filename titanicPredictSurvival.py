# Jon-Paul Boyd - Kaggle - Classifier to Predict Titantic Survial 
# Importing the libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import re as re


from sklearn.model_selection import cross_val_score
from hyperopt_dtc import Hyperopt_dtc
from hyperopt_rfc import Hyperopt_rfc
from hyperopt_gbc import Hyperopt_gbc
from hyperopt_xbc import Hyperopt_xbc
from hyperopt_knn import Hyperopt_knn


random_state = 0

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def set_age_master(input_dataset):
    for index, row in input_dataset.iterrows():
        if 'Master' in row['Title'] and math.isnan(row['Age']):
            input_dataset.at[index, 'Age'] = 11

    return input_dataset


# Importing the datasets
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
full_data = [dataset_train, dataset_test]


# Distribution Analysis on training set
print('_'*80, "=== Training set info ===", sep='\n')
print(dataset_train.columns.values, '_'*80, sep='\n')
print(dataset_train.info(), '_'*80, sep='\n')
print(dataset_train.head(), '_'*80, sep='\n')
print(dataset_train.tail(), '_'*80, sep='\n')
print(dataset_train.describe(), '_'*80, sep='\n')
print(dataset_train.describe(include=['O']), '_'*80, sep='\n')

print(dataset_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')
print(dataset_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')
print(dataset_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')
print(dataset_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')
print(dataset_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')

grid = sns.FacetGrid(dataset_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

grid = sns.FacetGrid(dataset_train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

grid = sns.FacetGrid(dataset_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()


# Feature Engineering
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print(dataset_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(dataset_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(dataset_train['Title'], dataset_train['Sex']))


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', \
                                                 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(dataset_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

dataset_train = set_age_master(dataset_train)
dataset_train[['Embarked']] = dataset_train[['Embarked']].fillna('S')
dataset_test = set_age_master(dataset_test)
dataset_test[['Embarked']] = dataset_test[['Embarked']].fillna('S')

dataset_train = pd.get_dummies(dataset_train, columns=['Pclass', 'Sex', 'Embarked', 'Title'], drop_first=True)
dataset_test = pd.get_dummies(dataset_test, columns=['Pclass', 'Sex', 'Embarked', 'Title'], drop_first=True)


X_train = dataset_train.iloc[:, [3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].values
y_train = dataset_train.iloc[:, 1].values
X_test = dataset_test.iloc[:,  [2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_train = imputer_train.fit(X_train[:, 0:1])
X_train[:, 0:1] = imputer_train.transform(X_train[:, 0:1])

imputer_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_test = imputer_test.fit(X_test[:, 0:1])
X_test[:, 0:1] = imputer_test.transform(X_test[:, 0:1])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
sc_test = StandardScaler()
X_train = sc_train.fit_transform(X_train)
X_test = sc_test.fit_transform(X_test)


# Hyperparameter optimisation, fit and score
hyperopt_dtc = Hyperopt_dtc(X_train, y_train)
dtc, best_params, best_dtc_acc = hyperopt_dtc.best()
print("DTC best acc ", best_dtc_acc)
y_pred_dtc = dtc.predict(X_test)

hyperopt_rfc = Hyperopt_rfc(X_train, y_train)
rfc, best_params, best_rfc_acc = hyperopt_rfc.best()
print("RFC best acc ", best_rfc_acc)
y_pred_rfc = rfc.predict(X_test)

hyperopt_gbc = Hyperopt_gbc(X_train, y_train)
gbc, best_params, best_gbc_acc = hyperopt_gbc.best()
print("GBC best acc ", best_gbc_acc)
y_pred_gbc = gbc.predict(X_test)

hyperopt_xbc = Hyperopt_xbc(X_train, y_train)
xbc, best_params, best_xbc_acc = hyperopt_xbc.best()
print("XBC best acc ", best_xbc_acc)
y_pred_xbc = xbc.predict(X_test)

hyperopt_knn = Hyperopt_knn(X_train, y_train)
knn, best_params, best_knn_acc = hyperopt_knn.best()
print("KNN best acc ", best_knn_acc)
y_pred_knn = knn.predict(X_test)


models = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XBC', 'KNN'],
    'Acc': [best_dtc_acc, best_rfc_acc, best_gbc_acc, best_xbc_acc, best_knn_acc]})
models.sort_values(by='Acc', ascending=True)
print(models)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_train, y_pred)
# Accuracy = correct divided by total
#accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])


submission = pd.DataFrame({
        "PassengerId": dataset_test["PassengerId"],
        "Survived": y_pred_xbc
    })

submission.to_csv('test_set_prediction.csv', index=False)
