import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,f1_score

pima = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

print(pima.head(10))
print(pima.shape)
print(pima.isnull().any())
pima.info()
X = pima.drop('Outcome', axis = 1)
y = pima['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

std = StandardScaler()

X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

rfcl = RandomForestClassifier()
rfcl.fit(X_train,y_train)

y_pred_rfcl = rfcl.predict(X_test)

svc = svm.SVC()
svc.fit(X_train,y_train)

y_pred_svc = svc.predict(X_test)
print(accuracy_score(y_pred_rfcl,y_test))  
print(accuracy_score(y_pred_svc,y_test))
print(confusion_matrix(y_pred_rfcl,y_test))  
print(confusion_matrix(y_pred_svc,y_test))
print(f1_score(y_pred_rfcl,y_test))
print(f1_score(y_pred_svc,y_test))
print(classification_report(y_pred_rfcl,y_test)) 
print(classification_report(y_pred_svc,y_test))
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error

wdata = pd.read_csv("../input/winedata/winequality_red.csv")

print(wdata.head(10))
print(wdata.shape)
print(wdata.isnull().any())
X = wdata.drop('quality', axis =1)
y = wdata['quality']

std = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

lr = LinearRegression()
rfr = RandomForestRegressor()

lr.fit(X_train,y_train)
rfr.fit(X_train,y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rfr = rfr.predict(X_test)
print("Mean Squared Error: ", mean_squared_error(y_pred_lr,y_test))  
print("Mean Squared Error: ",mean_squared_error(y_pred_rfr,y_test)) 
print("Mean Absolute Error: ",mean_absolute_error(y_pred_lr,y_test))
print("Mean Absolute Error",mean_absolute_error(y_pred_rfr,y_test)) 
Classification Accuracy = correct predictions/ total predictions
accuracy = total correct predictions / total predictions made * 100
interval = z * sqrt( (error * (1 - error)) / n)
interval = z * sqrt( (accuracy * (1 - accuracy)) / n)
# split the data into a train and validation sets
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5)
base_prediction = base_model.predict(X2)
error = mean_squared_error(base_prediction, y2) ** 0.5
mean = base_model.predict(X_test)
st_dev = error
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5)
base_model.fit(X1, y1)
base_prediction = base_model.predict(X2)
validation_error = (base_prediction - y2) ** 2
error_model.fit(X2, validation_error)
mean = base_model.predict(X_test)
st_dev = error_model.predict(X_test)