import numpy as np
import sys
sys.stdout = open('log.txt', 'w')
np.set_printoptions(threshold = np.inf)

class Perceptron:
    def __init__(this):
        this.lr = 0.1
        this.weights = None

    def Train_weights(this, data, data_labels):
        dimensions = len(data[0]) # get the number of features/dimensions of our data
        this.weights = np.random.normal(0, 0.1, dimensions) * 0.1 # initialize the weights for each feature/dimention to 0
        for indx, data_i in enumerate(data):
            linear_output = np.dot(this.weights, data_i) # f(w, x) = <w, x>
            class_predicted = this.Activation_function(linear_output)
            # Perceptron update rule
            update = this.lr * (data_labels[indx] - class_predicted)
            this.weights = this.weights + (update * data_i)
            #this.lr = this.lr * 0.9995 
        return this.weights       
    
    def Activation_function(this, linear_output):
        unit_step_function = np.where(linear_output >= 0, 1, 0)
        return unit_step_function
    
    def Predict(this, data_i):
        linear_output = np.dot(data_i, this.weights)
        label_predicted = this.Activation_function(linear_output)
        return label_predicted

    def Mean_absolute_error(this, data, data_labels):
        sum = 0
        n = len(data)
        for i in range(n):
            pred_i = this.Predict(data[i]) 
            sum += np.abs(pred_i - data_labels[i])
        return sum / n
    
    # Never used, yields the same results as Mean_absolute_error
    def Mean_squared_error(this, data, data_labels):
        sum = 0
        n = len(data)
        for i in range(n):
            pred_i = this.Predict(data[i]) 
            sum += np.power((pred_i - data_labels[i]), 2)
        return sum / n