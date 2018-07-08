# Jon-Paul Boyd - Kaggle - Classifier to Predict Titantic Survial 
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re as re
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from hyperopt_dtc import Hyperopt_dtc
from hyperopt_rfc import Hyperopt_rfc
from hyperopt_gbc import Hyperopt_gbc
from hyperopt_xbc import Hyperopt_xbc
from hyperopt_knn import Hyperopt_knn

seed = 1807
default_survival = 0.5

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


# Importing the datasets
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
dataset_full = dataset_train.append(dataset_test)


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
print('_'*80)

# Feature Engineering
dataset_full['FamilySize'] = dataset_full['SibSp'] + dataset_full['Parch'] + 1
print(dataset_full[['FamilySize', 'Survived']][:891].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')

dataset_full['IsAlone'] = 0
dataset_full.loc[dataset_full['FamilySize'] == 1, 'IsAlone'] = 1
print(dataset_full[['IsAlone', 'Survived']][:891].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False), '_'*80, sep='\n')

dataset_full['Title'] = dataset_full['Name'].apply(get_title)
print(pd.crosstab(dataset_full['Title'][:891], dataset_full['Sex'][:891]), '_'*80, sep='\n')


dataset_full['Title'] = dataset_full['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', \
                                                 'Jonkheer', 'Dona'], 'Rare')
dataset_full['Title'] = dataset_full['Title'].replace('Mlle', 'Miss')
dataset_full['Title'] = dataset_full['Title'].replace('Ms', 'Miss')
dataset_full['Title'] = dataset_full['Title'].replace('Mme', 'Mrs')

print(dataset_full[['Title', 'Survived']][:891].groupby(['Title'], as_index=False).mean(), '_'*80, sep='\n')

titles = ['Master', 'Miss', 'Mr', 'Mrs', 'Rare']
for title in titles:
    age_to_impute = dataset_full.groupby('Title')['Age'].median()[titles.index(title)]
    dataset_full.loc[(dataset_full['Age'].isnull()) & (dataset_full['Title'] == title), 'Age'] = age_to_impute
    
dataset_full['AgeBin'] = pd.qcut(dataset_full['Age'], 4)

label = LabelEncoder()
dataset_full['AgeBin_Code'] = label.fit_transform(dataset_full['AgeBin'])

dataset_full[['Embarked']] = dataset_full[['Embarked']].fillna('S')
dataset_full = pd.get_dummies(dataset_full, columns=['Pclass', 'Sex', 'Title', 'Embarked'], drop_first=True)


dataset_full['Last_Name'] = dataset_full['Name'].apply(lambda x: str.split(x, ",")[0])
dataset_full['Fare'].fillna(dataset_full['Fare'].mean(), inplace=True)


dataset_full['Family_Survival'] = default_survival

for grp, grp_df in dataset_full[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                dataset_full.loc[dataset_full['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                dataset_full.loc[dataset_full['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      dataset_full.loc[dataset_full['Family_Survival']!=0.5].shape[0])

for _, grp_df in dataset_full.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    dataset_full.loc[dataset_full['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    dataset_full.loc[dataset_full['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(dataset_full[dataset_full['Family_Survival']!=0.5].shape[0]))


dataset_full.drop(['Name', 'PassengerId', 'Fare', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Age', 'AgeBin', 'Last_Name'], axis = 1, inplace = True)
    

X_train_full = dataset_full[:891]
y_train = dataset_train.iloc[:, 1].values
X_train = X_train_full.drop(['Survived'], axis = 1)
X_test_full = dataset_full[891:]
X_test = X_test_full.drop(['Survived'], axis = 1)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
sc_test = StandardScaler()
X_train = sc_train.fit_transform(X_train)
X_test = sc_test.fit_transform(X_test)


models = pd.DataFrame(columns=['Name', 'Short', 'Score', 'Params', 'Clf', 'Pred'])
i = 0


# Hyperparameter optimisation, fit and score
hyperopt_dtc = Hyperopt_dtc(X_train, y_train, seed)
clf, name, name_short, best_params, score = hyperopt_dtc.best()
y_pred = clf.predict(X_test)
models.loc[i] = pd.Series(dict(
            Name=name, Short=name_short, Score=score, Params=best_params, Clf=clf, Pred=y_pred
        ))
i += 1

hyperopt_rfc = Hyperopt_rfc(X_train, y_train, seed)
clf, name, name_short, best_params, score = hyperopt_rfc.best()
y_pred = clf.predict(X_test)
models.loc[i] = pd.Series(dict(
            Name=name, Short=name_short, Score=score, Params=best_params, Clf=clf, Pred=y_pred
        ))
i += 1

hyperopt_gbc = Hyperopt_gbc(X_train, y_train, seed)
clf, name, name_short, best_params, score = hyperopt_gbc.best()
y_pred = clf.predict(X_test)
models.loc[i] = pd.Series(dict(
            Name=name, Short=name_short, Score=score, Params=best_params, Clf=clf, Pred=y_pred
        ))
i += 1

hyperopt_xbc = Hyperopt_xbc(X_train, y_train, seed)
clf, name, name_short, best_params, score = hyperopt_xbc.best()
y_pred = clf.predict(X_test)
models.loc[i] = pd.Series(dict(
            Name=name, Short=name_short, Score=score, Params=best_params, Clf=clf, Pred=y_pred
        ))
i += 1

hyperopt_knn = Hyperopt_knn(X_train, y_train, seed)
clf, name, name_short, best_params, score = hyperopt_knn.best()
y_pred = clf.predict(X_test)
models.loc[i] = pd.Series(dict(
            Name=name, Short=name_short, Score=score, Params=best_params, Clf=clf, Pred=y_pred
        ))
i += 1


# Output model results
models.sort_values(by=['Score'], ascending=False, inplace=True)
for index, row in models.iterrows():
    print(row['Score'], row['Name'], row['Short'], row['Params'])


# Making the Confusion Matrix
#cm = confusion_matrix(Y_train, y_pred)
# Accuracy = correct divided by total
#accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])


# Output best prediction
submission = pd.DataFrame({
        "PassengerId": dataset_test["PassengerId"],
        "Survived": models['Pred'].iloc[0]
    })

submission.to_csv('test_set_prediction.csv', index=False)
