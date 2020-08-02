import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 


dataset = pd.read_csv("weathercsv/weather.csv")
print(dataset.shape)
# dataset.plot(x='MinTemp', y='MaxTemp', style='o')
# plt.title('MinTemp vs MaxTemp')  
# plt.xlabel('MinTemp')  
# plt.ylabel('MaxTemp')  
# plt.show()
X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) #2d array 



def line_function_hx(theta0,theta1,x):
    print(x)
    print("h(x) = " + str(theta1*x+theta0))
    return theta1*x+theta0

def linear_reg(X_train,y_train):
    theta1,theta0 = 1,1
    alpha = 0.03
    epoch = 0 
    l = (X_train.shape)[0]
    while(epoch < 20):
        count = 0
        sum0,sum1 = 0,0
        while(count < l):
            sum0 = sum0 + (line_function_hx(theta0,theta1,X_train[count][0]) - y_train[count][0])
            sum1 = sum1 + (line_function_hx(theta0,theta1,X_train[count][0]) - y_train[count][0]) * X_train[count][0]
            count+=1
        print(sum0,sum1)

        theta0 = theta0 - (alpha*(sum0/l))
        theta1 = theta1 - (alpha*(sum1/l))
        print(theta0,theta1)
        epoch+=1


    return(theta0,theta1)

print(linear_reg(X_train,y_train))
print(f"X = ")