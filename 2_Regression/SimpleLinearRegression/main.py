### Simple Linear Regression 

## Importing the Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

## Importing the Dataset 
df = pd.read_csv('Salary_Data.csv')
# print(df)
X = df.iloc[: , : -1].values
y = df.iloc[: , -1].values

# print(X) # Independant variable
# print(y) # Depend variable


## Split the dataset into the Training Set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 0)
# print(X_train) 
# print(X_test)
# print(y_train)
# print(y_test)


## Training the SimpleLinear Regression Model on the Training set

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)



## Predicting the Test set result

y_pred = regressor.predict(X_test)
# print(y_pred)  # These are the predicted salaries for the Test Set


## Visualising the Training Set Result 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Years of Experience VS Salary (Training Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


## Visualising the Test Set Result
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train,regressor.predict(X_train) , color='red' )
plt.xlabel("Years of Experiencee vs Salary (Test Set)")
plt.ylabel("Salary")
plt.show()