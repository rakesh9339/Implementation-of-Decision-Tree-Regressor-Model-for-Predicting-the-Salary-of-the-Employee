# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data

    -Collect and clean data on employee salaries and features
    -Split data into training and testing sets

2.Define your model

    -Use a Decision Tree Regressor to recursively partition data based on input features
    -Determine maximum depth of tree and other hyperparameters

3.Train your model

    -Fit model to training data
    -Calculate mean salary value for each subset

4.Evaluate your model

    -Use model to make predictions on testing data
    -Calculate metrics such as MAE and MSE to evaluate performance

5.Tune hyperparameters

    -Experiment with different hyperparameters to improve performance

6.Deploy your model

    Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RAKESH JS
RegisterNumber:  212222230115
*/
```
```py
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

### Initial dataset:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707363/8e9d8bd8-2e1c-4317-8ba7-2e3d8486a1ee)


### Data Info:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707363/b06bb6dd-2fee-4c07-b167-7d7107f71624)


### Optimization of null values:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707363/dc78293b-837d-4796-9175-578dbf7c0fae)


### Converting string literals to numericl values using label encoder:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707363/c7ba6e49-afbe-4437-a1d7-ad49e24eacaf)


### Assigning x and y values:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707363/31adfe81-abe0-425f-af35-aceab7db96be)


### Mean Squared Error:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707363/ee0628e7-59f1-48e2-906b-deb3287164b4)


### R2 (variance):
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707363/97ddf085-4c5f-48ce-92f4-89e6ec034639)


### Prediction:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707363/ffa14091-980a-4124-a2de-ab3876a10274)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
