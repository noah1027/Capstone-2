import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#imports main dataset used for modeling and prints some info about the dataframe
masterframe = pd.read_csv(r'C:\Users\tveva\OneDrive\Desktop\ShowNTell\data\mlmasterset.csv', index_col=0)
pd.set_option("display.max_rows", None, "display.max_columns", None)


print(masterframe.head(10))
print(masterframe.columns)
print(masterframe.corr())

#initializes and separates the variables by "x"  and "y"
independentvariables = masterframe[[ 'unemploymentrate', 'effectivefederalfunds', 'spotoilbarrelprice', 'consumersentimentindex', 'housingbuildingpermits']]
dependentvariable = masterframe['recessionindicator']

#splits the dataset into a training set and a testing set
x_train, x_test, y_train, y_test = train_test_split(independentvariables, dependentvariable, test_size=0.35)

#instance of a logistic model
logitmodel = LogisticRegression(max_iter=1000)

#fits the model based on training data set
logitmodel.fit(x_train, y_train)

predictions = logitmodel.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print("[[True Positive, False Positive]\n[False Negative, True Negative]]")