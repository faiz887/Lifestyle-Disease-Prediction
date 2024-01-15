# -*- coding: utf-8 -*-
"""Project - Diabetes

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DQlfAvHJjiQUwlJD00htg5gBruFHa2-p

# Diabetes Prediction

***Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.***

# IMPORTING THE LIBRARIES
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

"""# LOADING THE DATASET"""

data = pd.read_csv('/content/diabetes.csv')

data.head()  #displaying the head of dataset

data.describe()      #description of dataset

data.info()

data.shape    #768 rows and 9 columns

data.value_counts()

data.dtypes

data.columns

"""***Checking Null Values***"""

data.isnull().sum()

data.isnull().any()

data.isnull().all()

"""# Exploratory Data Analysis"""

data.corr()

plt.figure(figsize = (12,10))
sns.heatmap(data.corr(), annot =True)

data.hist(figsize=(18,12))
plt.show()

plt.figure(figsize=(14,10))
sns.set_style(style='whitegrid')
plt.subplot(2,3,1)
sns.boxplot(x='Glucose',data=data)
plt.subplot(2,3,2)
sns.boxplot(x='BloodPressure',data=data)
plt.subplot(2,3,3)
sns.boxplot(x='Insulin',data=data)
plt.subplot(2,3,4)
sns.boxplot(x='BMI',data=data)
plt.subplot(2,3,5)
sns.boxplot(x='Age',data=data)
plt.subplot(2,3,6)
sns.boxplot(x='SkinThickness',data=data)

mean_col = ['Glucose','BloodPressure','Insulin','Age','Outcome','BMI']
sns.pairplot(data[mean_col],palette='Accent')

sns.boxplot(x='Outcome',y='Insulin',data=data)

sns.regplot(x='BMI', y= 'Glucose', data=data)

sns.relplot(x='BMI', y= 'Glucose', data=data)

sns.scatterplot(x='Glucose', y= 'Insulin', data=data)

sns.jointplot(x='SkinThickness', y= 'Insulin', data=data)

sns.pairplot(data,hue='Outcome')

sns.lineplot(x='Glucose', y= 'Insulin', data=data)

sns.swarmplot(x='Glucose', y= 'Insulin', data=data)

sns.barplot(x="SkinThickness", y="Insulin", data=data[170:180])
plt.title("SkinThickness vs Insulin",fontsize=15)
plt.xlabel("SkinThickness")
plt.ylabel("Insulin")
plt.show()
plt.style.use("ggplot")

plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Glucose", y="Insulin", data=data[170:180])
plt.title("Glucose vs Insulin",fontsize=15)
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.show()

"""# TRAINING AND TESTING DATA"""

#train_test_splitting of the dataset
x = data.drop(columns = 'Outcome')

# Getting Predicting Value
y = data['Outcome']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

"""# MODELS

# 1. Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",reg.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

"""**So we get a accuracy score of 82.46 % using Logistic Regression**

# 2. KNeighborsClassifier
"""

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",knn.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

"""**So we get a accuracy score of 75.97 % using KNeighborsClassifier**

# 3. SVC
"""

from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)

y_pred=svc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",svc.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

"""**So we get a accuracy score of 79.22 % using SVC**

# 4. Naive Bayes
"""

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",gnb.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print("Accuracy Score:\n",gnb.score(x_train,y_train)*100)

"""**So we get a accuracy score of 75.73 % using Naiye Bayes**

# 5. DECISION TREE CLASSIFIER
"""

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)

y_pred=dtree.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",dtree.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

"""**So we get accuracy score of 73.37 % using DecisionTreeClassifier**

# 6.  RandomForestClassifier
"""

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

"""**So we get a accuracy score of 81.18 % using RandomForestClassifier**

# Ensemble (Voting Classifier)
"""

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Create individual models
logistic = LogisticRegression()
knn = KNeighborsClassifier()
svm = SVC(probability=True)  # Note: probability=True is needed for Voting Classifier
naive_bayes = GaussianNB()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()

# Create a Voting Classifier
ensemble_classifier = VotingClassifier(
    estimators=[
        ('logistic', logistic),
        ('knn', knn),
        ('svm', svm),
        ('naive_bayes', naive_bayes),
        ('decision_tree', decision_tree),
        ('random_forest', random_forest)
    ],
    voting='hard'  # 'hard' for majority voting, 'soft' for weighted voting based on probabilities
)

# Train the Voting Classifier
ensemble_classifier.fit(x_train, y_train)

# Predictions on the testing set
ensemble_predictions = ensemble_classifier.predict(x_test)

# Evaluate the ensemble model
print("Classification Report for Ensemble Classifier:")
print(classification_report(y_test, ensemble_predictions))

# Accuracy score for the ensemble model
accuracy = accuracy_score(y_test, ensemble_predictions)
print("Accuracy for Ensemble Classifier:", accuracy*100)

import matplotlib.pyplot as plt

# List of models
models = ['Logistic Regression', 'KNeighbors Classifier', 'SVC', 'Naive Bayes', 'Decision Tree Classifier', 'Random Forest Classifier', 'Ensemble']

# Corresponding accuracy values
accuracy_values = [82.46, 75.97, 79.22, 75.73, 73.37, 81.57, 81.16]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(models, accuracy_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of Different Models')
plt.ylim(0, 100)  # Set the y-axis range from 0 to 100 for better visualization
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Display the accuracy values on top of the bars
for i, value in enumerate(accuracy_values):
    plt.text(i, value + 1, f'{value:.2f}%', ha='center', va='bottom')

# Show the plot
plt.tight_layout()
plt.show()

"""***So now we conclude the accuracy of different models:***

* Logistic Regression= 82.46 %
* KNeighbors Classifier= 75.97 %
* SVC= 79.22 %
* Naiye Bayes= 75.73 %
* Decision Tree Classifier= 73.37%
* Random Forest Classifier= 81.57%
* Ensemble (Voting Classifier) = 81.16%




"""

import pickle
# Save the ensemble model
with open('model.pkl', 'wb') as file:
    pickle.dump(ensemble_classifier, file)