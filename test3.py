#-------------------------------Relu and sigmoid --------------------------------


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_test.csv\mnist_test.csv')
data.head()

data  = np.array(data)
m, n = data.shape # rows, columns - dimension
np.random.shuffle(data)
print(data)


#Split the  data into dev and trainig sets to prevent from overfitting to one collection
# first 1000 eg from the 10,000 eg
data_dev = data[0:1000].T #transposing it so each column is now and eg
y_dev =data_dev[0]
x_dev = data_dev[1:n]

#remaning 1001 to 10,000 egs
data_train = data[1000:m].T 
Y_train = data_train[0]  # Takes care of all the Labels which are essentially giving the true value
X_train = data_train[1:n] # 1st pixel to the 784th pixel
X_train = X_train / 255 # All the values are in grey scale so dividing gives values less than 1

def init_params():
    #10=rows, 784=columns
    W1 = np.random.rand(10, 784) - 0.5 
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2
    
def ReLU(Z):
    return np.maximum(Z, 0)

#For probability
# def softmax(Z):
#     Val = np.exp(Z) 
#     return Val / sum(np.exp(Z))
def sigmoid(Z):
    # A = np.exp(Z - np.max(Z)) 
    # B = np.sum(A)
    # return A/B
    return 1/(1 + np.exp(-Z))
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2
    
# Converting the labels to the matrix 
# Labels gives the true values which is required for backprop so we can get the error
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Y.size gives no. of eg or m & max means 9+1=10
    one_hot_Y[np.arange(Y.size), Y] = 1 # For each column go to the specified Y or label and set it to 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0
#This works since when boolean converts to nums True =1 , False = 0
# so if u put 5 it is greater than 0 so true is op that becomes 1
# -5 is false which becomes 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    #m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def  update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2





def get_predictions(A2):
    # argmax returns index rather than th value
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        #Checks at multiples of 50
        if (i % 10 == 0):
                print("Iteration: ", i)
                predictions = get_predictions(A2)
                print("Accuracy: ", get_accuracy(predictions, Y))
    return W1, b1, W2, b2
    


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.10)

