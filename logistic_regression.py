import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve,f1_score,precision_score,recall_score,accuracy_score,roc_auc_score


def roc_auc_curve(y_true,y_pred):
    fpr,tpr,threshold = roc_curve(y_true, y_pred)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Ratio')
    plt.ylabel('True Positive Ratio')
    plt.show()
def precission_recall_curve_(y_true,y_pred):
    pre,rec,threshold = precision_recall_curve(y_true,y_pred)
    plt.plot(pre,rec)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()
    
def model_report(y_true,y_pred):
    print("accuracy =",accuracy_score(y_true,y_pred))
    print("f1 score =",f1_score(y_true, y_pred))
    print('precission =',precision_score(y_true, y_pred))
    print('recall =',recall_score(y_true, y_pred))
    print('Roc_Auc score =',roc_auc_score(y_true, y_pred))
    
def grid_result(grid):
    print('best score =',grid.best_score_)
    print('best parameters =',grid.best_params_)
df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:,2:4].values
y = df.iloc[:,4].values

'''from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:,3] = labelencoder_x.fit_transform(X[:,3])
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
X = X[:,1:X.shape[1]]
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#insert 1 in the first column for bias
X = np.insert(X,0,1,axis = 1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


def Sigmoid(z):
    return 1/(1 + np.e**(-z))

def ComputeCost(X,y,theta,lambda_):
    mm = len(X)
    hypothesis = Sigmoid(X.dot(theta))
    val1 = np.log(hypothesis)
    val2 = np.log(1-hypothesis)
    j = -((y.T).dot(val1) + (1 - y.T).dot(val2))/mm + ((theta[1:].T).dot(theta[1:]))*(lambda_/(2*mm))
    
    return j

def GradientDescent(X,y,theta,lambda_ = 10,alpha = 0.001,num_iters = 40000):
    mm = len(X)
    j_history = np.zeros([num_iters,1])
    y = y.reshape(X.shape[0],1)
    for i in range(num_iters):
        hypothesis = Sigmoid(X.dot(theta))
        value = hypothesis - y #(50,1)
        val = sum(X*value).reshape(-1,1)
        theta[0] = theta[0] - (val*(alpha/mm))[0]
        theta[1:] = theta[1:] - (val*(alpha/mm))[1:] - theta[1:]*(lambda_*alpha/mm)
        j_history[i] = ComputeCost(X,y,theta,lambda_)
        if j_history[i] <= 0.1:
            break
   
    return theta,j_history

def predict(X,theta):
    p = Sigmoid(X.dot(theta))
    p = pd.DataFrame(p)
    p[p>=0.5] = 1
    p[p<0.5] = 0
    return p

theta = np.zeros([X.shape[1],1])
alpha = 0.001 
lambda_ = 20
num_iters = 60000
theta,j_history = GradientDescent(X_train,y_train,theta,lambda_,alpha,num_iters)
#plotting cost function over iteration
plt.plot(j_history)
plt.xlabel("Number of Iteration")
plt.ylabel('J cost')
plt.title('Cost over iteration')
plt.show()

print("theta computed from the gradient descent ")
print(theta)

y_pred = predict(X_test,theta)

roc_auc_curve(y_test, y_pred)
model_report(y_test, y_pred)
precission_recall_curve_(y_test, y_pred)
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


