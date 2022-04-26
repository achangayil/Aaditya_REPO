#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, plot_confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# This project is a Credit Card Fradulent Checker. As there are two response outcome Fraud/Not Fraud, this is
# a Supervised Classification problem. Hence we will be using decision trees 
# (RandomForestClassifier and GradientBoostingClassifier ) and also Logistic Regression

# cc_data is a dataframe that stores the csv file. read_csv is a function from pandas which reads the csv file
cc_data = pd.read_csv("/Users/aadityachangayil/Downloads/Credit-card-dataset/creditcard.csv")

# head(6) prints the first 6 rows from the csv file
#print(cc_data.head(6))

# checks if any values are null in the dataset
cc_data.isnull().values.any()
# the describe method looks through the dataset and and outputs the statistical data of the specified group of data. 
# it specifies the column name and the statistics of the given column are outputted
cc_data["Amount"].describe()


# non_fraud stores the length of data values under the Class column where the value is equal to 0.
# It shows the non-fraud values
non_fraud = len(cc_data[cc_data.Class == 0])
# fraud stores the length of values under the class column equal to 1. It shows the fraud values
fraud = len(cc_data[cc_data.Class == 1])
# percentage of fraud transaction of the transactions
fraud_percent = (fraud / (fraud + non_fraud)) * 100

print("Number of Genuine transactions: ", non_fraud)
print("Number of Fraud transactions: ", fraud)
# The value is formatted using a special function. 
#{:.4f} represents the the number of places after the decimal point that will be outputted
print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))

labels = ["Genuine", "Fraud"] # Assign labels for the non_fraud and fraud values, respectively.
# The data values in class are counted, then sorted.
count_classes = cc_data.value_counts(cc_data['Class'], sort = True)
#plot function plots the graph. It can plot with special parameters. kind represents the type of graph, 
# which is bar, for bar graph, and rot is equal to 0 for no rotation
count_classes.plot(kind = "bar", rot = 0)
# Createt the title for the graph using plt.title function
plt.title("Visualization of Labels")
plt.ylabel("Count")  # xlabel and ylabel are pyplot functions that set labels for the y and x axis
# The xticks pyplot function is used to get and set the current tick locations and labels of the x axis 
plt.xticks(range(2), labels) 
# show() shows the diagram
plt.show()
# StandardScaler() removes the mean and scales, which is normalizing the features in a specific range to unit variance
scaler = StandardScaler()
# Column is created in cc_data. fit_transform function transforms the dataframe' column Amount's values into 
# 2-dimensional arrays using the reshape function and add a new column created with the scaled values.
cc_data["NormalizedAmount"] = scaler.fit_transform(cc_data["Amount"].values.reshape(-1, 1))
# The original Amount column and the Time column are dropped. axis = 1 represents columns
cc_data.drop(["Amount", "Time"], inplace= True, axis= 1)
Y = cc_data["Class"] # Class in dataframe cc_data is assigned as Y column
X = cc_data.drop(["Class"], axis= 1) # Class column is dropped from cc_data
# Assigning data for the train and test data variables using split function and random number generator
(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size= 0.2, random_state = 123)
print("Shape of train_X: ", train_X.shape)
print("Shape of test_X: ", test_X.shape)

# Decision Tree Classifier is a classification model which creates a decision tree. A Decision Tree is a tree which specifies
# choices (nodes) based on choices made. Random forest creates mumtiple trees and takes votes to decide.
decision_tree = DecisionTreeClassifier() 

# Random Forest Classifier is a classification model which creates a Random Forest, which is a classifier that contains
#subsets a given dataset given through decision trees. That is called the train data set.
random_forest = RandomForestClassifier(n_estimators = 100)

# fit method will build decision tree classifier from a given training set
decision_tree.fit(train_X, train_Y)
# predict function is used by the models to record predictions from train data
dt_predictions = decision_tree.predict(test_X)
# score method visualizes the accuracy percentage, which here represents scores of the credit card classifiers
decision_tree_score = decision_tree.score(test_X, test_Y) * 100
# fit metod is used here for building Random Forest classifier from a given training set
random_forest.fit(train_X, train_Y) 
# preiction function is used for Random Forest. It takes in the test_x dataset for the function.
predictions_rf = random_forest.predict(test_X)
random_forest_score = random_forest.score(test_X, test_Y) * 100
print("Random Forest Score: ", random_forest_score)
print("Decision Tree Score: ", decision_tree_score)

# metrics is a method name representing sklearn. actuals and predictions are passed as parameters.
def metrics(actuals, predictions):
    # These modules are imported sklearn. actuals and predictions are passed as them. They are formatted to have 5 decimal places after the decimal point.
    # accuracy_score measures the number of accurate predictions (True Positive and True negetive) out of all the predictions.   
    print("Accuracy: {:.5f}".format(accuracy_score(actuals, predictions)))
    # precision_score measures the amount of precise positive predictions (True Positive) out of all positive predictions.
    print("Precision: {:.5f}".format(precision_score(actuals, predictions)))
    # recall_score measures the amount of True positive predictions of actually positvive samples.
    print("Recall: {:.5f}".format(recall_score(actuals, predictions)))
    # f1_score method calculates the f1_score, the harmonic mean of precision and recall.
    # It is calculated by 2*(precision*recall)/(precision+recall)
    print("F1-score: {:.5f}".format(f1_score(actuals, predictions)))

# the confusion matrix function calculates the confusion matrix, which is the summary of prediction 
# results, correct and incorrect, made by a classifier. It helps give an insight into the errors made
# by a classifier as well as their type.
confusion_matrix_dt = confusion_matrix(test_Y, dt_predictions.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt)

# ConfusionMatrixDisplay plots the confusion matrix given the true and predicted labels
dt_disp = ConfusionMatrixDisplay(confusion_matrix_dt)
dt_disp.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()

print("Evaluation of Decision Tree Model")
print()
# calling the method and passing test_Y and dt_predictions.round() for Decision Tree
metrics(test_Y, dt_predictions.round())
# the confusion matrix method is called with test_Y passed in as actuals, and predictions_rf.round() passed in as predictions.
confusion_matrix_rf = confusion_matrix(test_Y, predictions_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)
rf_disp = ConfusionMatrixDisplay(confusion_matrix_rf)
rf_disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()

print("Evaluation of Random Forest Model")
print()
# calling metrics again, but this time for Random Forest
metrics(test_Y, predictions_rf.round())


# SMOTE (Synthetic Minority Oversampling TechniquE) is an oversampling method used for imbalanced 
# class distributions. It replicates minority class examples to randomly increase them, balancing the class distirbution.
# fit_resample() resamples the data, where it replicates the minority representation, and hence removes class imbalance
X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
print("Resampled shape of X: ", X_resampled.shape)
print("Resampled shape of Y: ", Y_resampled.shape)



value_counts = Counter(Y_resampled)
print(value_counts)
(train_X, test_X, train_Y, test_Y) = train_test_split(X_resampled, Y_resampled, test_size= 0.3, random_state= 42)

rf_resampled = RandomForestClassifier(n_estimators = 100)
rf_resampled.fit(train_X, train_Y)
predictions_resampled = rf_resampled.predict(test_X)
random_forest_score_resampled = rf_resampled.score(test_X, test_Y) * 100
y_predict = random_forest.predict(test_Y)
cm_resampled = confusion_matrix(test_Y, y_predict.round())
print("Confusion Matrix - Random Forest")
print(cm_resampled)

rf_disp2 = ConfusionMatrixDisplay(cm_resampled)
rf_disp2.plot()
plt.title("Confusion Matrix - Random Forest After Oversampling")
plt.show()


print("Evaluation of Random Forest Model")
print()
metrics(test_Y, predictions_resampled.round())


# In[6]:


#Gradient Boosted Model
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
# GBM, Random forest are all based on decision tree.
Hyperparameter training can be done when tunig the model. There are many hyperparameters but I am mentioning 
# the improtant ones like 
#learning_rate -- this parameter scales the contribution of each tree, 
# n_estimators -- the number of trees to construct, 
# subsample -- The maximum depth of each tree,  
# max_depth -- The maximum depth of each tree
# max_features -- This is max.
# We can apply different range of hypeparameters using a grid. 
gradient_booster = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, subsample=0.5,  max_features=10)
gradient_booster.get_params()


# In[15]:


import timeit
#This piece of code creates a `GBM model. As this time to create it, we are capturing the time it takes
start = timeit.timeit()
gradient_booster.fit(train_X,train_Y)
end = timeit.timeit()
print("--- Time it took to create a GBM Model")
end - start


# In[8]:


print(classification_report(test_Y,gradient_booster.predict(test_X)))


# In[9]:


from sklearn.linear_model import LogisticRegression
# Logistic Regression model is also called logit model, is a statistical model that models the probability 
# of one event (true/false) taking place by having the log-odds (the logarithm of the odds)
model_lr = LogisticRegression()
model_lr.fit(train_X, train_Y)
train_acc = model_lr.score(train_X, train_Y)
print("The Accuracy for Training Set is {}".format(train_acc*100))

#Evaluating a test set
y_pred = model_lr.predict(test_X)
test_acc = accuracy_score(test_Y, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc*100))
print(classification_report(test_Y, y_pred))


# In[10]:


# Seaborn is a package for visualization. It is used to create the confusion matrix figure
import seaborn as sns
cm=confusion_matrix(test_Y,y_pred)
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix")
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig('confusion_matrix.png')


# In[14]:


from sklearn import metrics
y_pred_proba = model_lr.predict_proba(test_X)[::,1]
fpr, tpr, _ = metrics.roc_curve(test_Y,  y_pred_proba)

# create ROC curve
# A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic 
# ability of a binary classifier system as its discrimination threshold is varied. 
# The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) 
# TPR (Recall) = total True positive/Total postives 
# FPR = = total True Negatives/Total Negatives 

plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




