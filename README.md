# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
data = pd.read_csv("D:/introduction to ML/jupyter notebooks/mama/Employee.csv")
data.head()
```
## output:
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/a0894abd-5425-4f4e-a8a5-afe6d8f06ab3)

```
data.info()
data.isnull().sum()
```
## output:
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/2d1870be-45ae-4072-94f2-457bd3fc7f8c)
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/cecd03aa-658d-4df6-b82f-8a1ef4ea0980)

```
data['left'].value_counts()
```
## output:
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/e0a14a4c-97b2-40e8-a8a9-fb19c107dd82)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
```
## output:
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/4119b636-8467-4173-867b-a83dec66b243)
```
y = data['left']
y.head()
```
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/0f00de90-8207-4a28-ac3c-bb47057ae205)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(x_train,y_train)
y_predict = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_predict)
accuracy
```
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/e1c1c5e3-9abd-4b70-ba69-af128b8e87d8)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/20d0366c-a7f9-4027-8976-733d7214b398)


```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Abdur Rahman Basil A H
RegisterNumber:  212223040002
*/
```

## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

