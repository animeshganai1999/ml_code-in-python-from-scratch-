import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('50_Startups.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:,3] = labelencoder_x.fit_transform(X[:,3])
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
X = X[:,1:X.shape[1]]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#insert 1 in the first column for bias
X = np.insert(X,0,1,axis = 1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

def ComputeCost(X,y,theta,lambda_ = 10):
    m = len(X)
    hypothesis = X.dot(theta)
    value = hypothesis - y.reshape(X.shape[0],1)
    value = (sum(value**2))/(2*m) + (theta[1:].T).dot(theta[1:])*(lambda_/(2*m))
    return value

def GradientDescent(X,y,theta,lambda_ = 10,alpha = 0.001,num_iters = 40000):
    m = len(X)
    j_history = np.zeros([num_iters,1])
    y = y.reshape(X.shape[0],1)
    for i in range(num_iters):
        hypothesis = X.dot(theta)
        
        value = hypothesis - y 
        val = sum(X*value).reshape(-1,1)
        theta[0] = theta[0] - (val*(alpha/m))[0]
        theta[1:] = theta[1:] - (val*(alpha/m))[1:] - theta[1:]*((alpha*lambda_)/m)
        j_history[i] = ComputeCost(X,y,theta,lambda_)
        if j_history[i] <= 0.1:
            break
   
    return theta,j_history

def predict(X,theta):
    return X.dot(theta)

theta = np.zeros([X.shape[1],1])
alpha = 0.001 
num_iters = 60000
lambda_ = 20
theta,j_history = GradientDescent(X_train,y_train,theta,lambda_,alpha,num_iters)
#plotting cost function over iteration
plt.plot(j_history)
plt.xlabel("Number of Iteration")
plt.ylabel('J cost')
plt.title('Cost over iteration')
plt.show()

print("theta computed from the gradient descent ")
print(theta)

y_pred = predict(X_test, theta)

plt.plot(y_test,color = 'blue',label = 'test value')
plt.plot(y_pred,color = 'green',label = 'predicted value')
plt.legend()
plt.show()

