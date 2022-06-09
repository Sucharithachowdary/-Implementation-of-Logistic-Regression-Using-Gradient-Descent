# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner
``
## Algorithm
## step1:
Use the standard libraries in python for finding linear regression.
## step2:
Set variables for assigning dataset values
## step3:
Import linear regression from sklearn.
## step4:
Predict the values of array
## step5:
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
## step6:
Obtain the graph.
## step7:
stop the programme.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:k sucharitha
RegisterNumber:212221240021  

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

datasets=pd.read_csv('/content/Social_Network_Ads (1).csv')
x=datasets.iloc[:,[2,3]].values
y=datasets.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_x
StandardScaler()

x_train=sc_x.fit_transform(x_train)
x_Test=sc_x.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_Pred = classifier.predict(x_test)
y_Pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_Pred)
cm

from sklearn import metrics
accuracy =metrics.accuracy_score(y_test,y_Pred)
accuracy

recall_sensitivity=metrics.recall_score(y_test,y_Pred,pos_label=1)
recall_specificity=metrics.recall_score(y_test,y_Pred,pos_label=0)
recall_sensitivity,recall_specificity

from matplotlib.colors import ListedColormap
x_Set,y_Set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_Set[:,0].min()-1,stop=x_Set[:,0].max()+1,step=0.01),np.arange(start=x_Set[:,1].min()-1,stop=x_Set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_Set)):
  plt.scatter(x_Set[y_Set==j,0],x_Set[y_Set==j,1],c=ListedColormap(('red','green'))(i),label=j)
  plt.title('LogisticRegression(Trainingset)')
  plt.xlabel('Age')
  plt.ylabel('Estimated Salary')
  plt.legend()
  plt.show()
*/
```
## Output:
![4 1](https://user-images.githubusercontent.com/94166007/172888308-355f10aa-8c49-4b44-a5c8-01fb055a7541.jpeg)

![4 2](https://user-images.githubusercontent.com/94166007/172888451-6d620b91-3804-478e-8ef5-82b09587063b.jpeg)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


