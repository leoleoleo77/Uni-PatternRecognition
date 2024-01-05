import dataMethods
import numpy as np
import sys
sys.stdout = open('log.txt', 'w')
np.set_printoptions(threshold = np.inf)

class Perceptron:
    def __init__(this, learning_rate=0.0001):
        this.lr = learning_rate
        this.weights = None

    def Train_weights(this, data, data_labels):
        n_data = len(data) # the number of data the algorithm trains on
        dimensions = len(data[0]) # get the number of features/dimensions of our data
        this.weights = np.random.normal(0, 0.1, dimensions) * 0.1 #np.zeros(dimensions) # initialize the weights for each feature/dimention to 0
        
        current_accurancy = this.Accurancy(data, data_labels)
        print("Starting Accuracy: ", current_accurancy)
        print("Starting Weights: ", this.weights)
        
        while n_data > 0:
            for indx, data_i in enumerate(data):
                linear_output = np.dot(this.weights, data_i) # f(w, x) = <w, x>
                class_predicted = this.Activation_function(linear_output)
                # Perceptron update rule
                update = this.lr * (data_labels[indx] - class_predicted)
                this.weights = this.weights + (update * data_i)
            current_accurancy = this.Accurancy(data, data_labels)
            print("Accuracy: ", current_accurancy)
            #sprint("Weights: ", this.weights)

            n_data -= 1
        return this.weights
    
    def Activation_function(this, linear_output):
        unit_step_function = np.where(linear_output >= 0, 1, 0)
        return np.array(unit_step_function)
    
    def Predict(this, data_i):
        linear_output = np.dot(data_i, this.weights)
        label_predicted = this.Activation_function(linear_output)
        return label_predicted

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
mydata = dataMethods.Perceptron_data()
training_data = mydata.Training()[:-17000]
training_labels = mydata.Training_labels()[:-17000]
print(p.Train_weights(training_data, training_labels))
print(p.Accurancy(mydata.Testing(), mydata.Testing_labels()))
