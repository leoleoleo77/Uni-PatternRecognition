import numpy as np

class Least_Squares_Method:
        def __init__(this):
                this.weights = []
        
        def Train(this, data, data_labels):
                X = data
                y = data_labels

                X_T = np.transpose(X)
                XTX = np.float64(np.dot(X_T, X))
                XTX_inv = np.linalg.inv(XTX)
                XTy = np.dot(X_T, y)

                this.weights = np.dot(XTX_inv, XTy)

        def Predict(this, data_i):
                linear_output = np.dot(data_i, this.weights)
                return linear_output
        
        def Mean_absolute_error(this, data, data_labels):
                sum = 0
                n = len(data)
                for i in range(n):
                        pred_i = this.Predict(data[i]) 
                        sum += np.abs(pred_i - data_labels[i])
                return sum / n
    
        def Mean_squared_error(this, data, data_labels):
                sum = 0
                n = len(data)
                for i in range(n):
                        pred_i = this.Predict(data[i]) 
                        sum += np.power((pred_i - data_labels[i]), 2)
                return sum / n



# ⇊⇊⇊ COOL STUFF BUT NOT NEEDED ⇊⇊⇊
def Gradient_descent_Training(data, data_labels, lr=0.01):
        dimensions = len(data[0]) # get the number of features/dimensions of our data
        weights = np.random.normal(0, 0.1, dimensions) * 0.1# initialize the weights for each feature/dimention to 0
        n = len(data)
        for i in range(n):
                linear_output = np.dot(weights, data[i]) # f(w, x) = <w, x>
                update = lr * (data_labels[i] - linear_output)
                weights = weights + update * data[i]
        return weights

