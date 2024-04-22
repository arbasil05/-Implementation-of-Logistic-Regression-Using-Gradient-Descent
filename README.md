# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### 1. Load the Dataset
### 2. Preprocess the Data
### 3. Split the Data
### 4. Build and Train the Model
### 5.Evaluate the Model
### 6.Predict New Data

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("D:/introduction to ML/jupyter notebooks/Placement_Data.csv")
dataset

#dropping the serial no and salary col

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset
```

### Output:
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/ee088f29-4c49-4480-ac76-bc4503f5bdcb)


## Program:
```
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
```
### Output:
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/dd92a9d8-d7da-4791-b82f-ffeda66362cc)

## Program:
```
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

dataset

```
### Output:



## Program:
```

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
#Initialize the model parameters
theta = np.random.randn(X.shape[1])
y = Y
#Define the sigmoid function:
def sigmoid(z):
    return 1 / (1+np.exp(-z))
# define the loss function:
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

#define the gradient descent algorithms
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y)/m
        theta -= alpha*gradient
    return theta

#train the model:
theta = gradient_descent(theta,X,y,alpha = 0.01,num_iterations = 1000)

# make predictions:
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5,1,0)
    return y_pred

y_pred = predict(theta,X)
y_pred

Accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy: ",Accuracy)
```
### Output:
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/4580c924-1e59-4a1b-aef9-af0fc9a99678)

## Program:
```
print(y_pred)
```
### Output:
![image](https://github.com/arbasil05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144218037/69152acc-bbef-4fdd-8bd6-bbe77f11785f)


## Program:
```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```




/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Abdur Rahman Basil A H
RegisterNumber:  212223040002
*/


## Output:

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

