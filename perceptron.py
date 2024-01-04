
import dataMethods
import numpy as np
import sys
sys.stdout = open('log.txt', 'w')
np.set_printoptions(threshold = np.inf)

class Perceptron:
    def __init__(this, learning_rate=0.01):
        this.lr = learning_rate
        this.weights = None

    def Train_weights(this, data, data_labels):
        n_data = len(data) # the number of data the algorithm trains on
        dimensions = len(data[0]) # get the number of features/dimensions of our data
        this.weights = np.zeros(dimensions) # initialize the weights for each feature/dimention to 0
        #print(data)
        while n_data > 0:

            current_accurancy = this.Accurancy(data, data_labels)
            print("Weights: ", this.weights)
            print("Accuracy: ", current_accurancy)

            for indx, data_i in enumerate(data):
                linear_output = np.multiply(this.weights, data_i) # f(w, x) = <w, x>
                class_predicted = this.Activation_function(linear_output)
                # Perceptron update rule
                print(data_labels[indx])
                update = this.lr * (data_labels[indx] - class_predicted)
                this.weights = this.weights + (update * data_i)
            n_data -= 1
        return this.weights
    
    def Activation_function(this, linear_output):
        # unit_step_function = np.where(x.any() >= 0, 1, 0)
        unit_step_function = []
        for x in linear_output:
            if type(x) == float:
                if x >= 0:
                    unit_step_function.append(1)
                else:
                    unit_step_function.append(0)
            else:
                unit_step_function.append(1)
        return np.array(unit_step_function)
    
    def Predict(this, data_i):
        activation = np.dot(data_i, this.weights)
        '''activation = 0.0
        for input, weight in zip(data_i, this.weights):
            if type(input) == float:
                activation += input * weight
        if activation >= 0.0:
            return 1.0 '''
        return this.softmax(activation)
    
    def softmax(this, x):
        exp_values = np.exp(x - np.max(x))  # For numerical stability
        return exp_values / np.sum(exp_values, axis=0)

    def Accurancy(this, data, data_labels):
        correct_predictions = 0.0
        predictions = []
        for i in range(len(data)):
            pred_i = this.Predict(data[i]) 
            predictions.append(pred_i)
            if pred_i == data_labels[i]:
                correct_predictions += 1.0
        return correct_predictions / float(len(data))
    
p = Perceptron()
mydata = dataMethods.Perceptron_data(0.01)
training_data = mydata.Training()
training_labels = mydata.Training_labels()
print(p.Train_weights(training_data, training_labels))