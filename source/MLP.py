import numpy as np
import sys
sys.stdout = open('log.txt', 'w')
np.set_printoptions(threshold = np.inf)

class MultiLayerPerceptron:
    def __init__(this):
        this.n_nodes = 10 # Number of nodes inside the hidden layer
        this.hl_weights = [] # hidden layer weights
        this.out_weights = None # output weights

    def Train(this, data, data_labels):
        this.lr = 0.00001
        dimensions = len(data[0]) # get the number of features/dimensions of our data
        for _ in range(this.n_nodes):
            this.hl_weights.append(np.random.normal(0, 0.1, dimensions) * 1) # initialize the weights for each feature/dimention to 0
        this.out_weights = np.random.normal(0, 0.1, this.n_nodes) * 1 # init the output weights 
        #this.out_weights = np.ones(dimensions)
        for indx, data_i in enumerate(data): # for each datapoint
            Z = []
            # Hidden layer nodes ↓
            output = 0
            for i in range(this.n_nodes): # for each node inside the hidden layer
                linear_output = np.dot(this.hl_weights[i], data_i) # f(w, x) = <w, x>
                node_i = this.Leaky_ReLU(linear_output)
                Z.append(node_i)
                output += np.dot(this.out_weights[i], node_i)

            # Output node ↓
            class_predicted = this.Sigmoid(output)
            # Binary Cross-Entropy Loss ...ㄟ( ▔, ▔ )ㄏ
            loss = this.log_loss(class_predicted, data_labels[indx])
            # loss = data_labels[indx] - class_predicted

            # Below is the update of the weights (Backpropagation?)
            # I used the article below as a reference to a great extent.
            # https://builtin.com/machine-learning/backpropagation-neural-network
            # Calculate deltas
            delta = []
            for i in range(this.n_nodes):
                delta_i = this.out_weights[i] * loss * this.ReLU_derevative(Z[i])
                delta.append(delta_i)
            for i in range(this.n_nodes):
                # update output weights
                this.out_weights[i] += - this.lr * Z[i] * loss
                # update hidden layer weights
                this.hl_weights[i] = this.hl_weights[i] - this.lr * data_i * delta[i]
            
            # learning rate decay
            this.lr = this.lr * np.exp(-0.000001*indx)

    def Predict(this, data_i):
        output = 0
        for i in range(this.n_nodes): # for each node inside the hidden layer
            linear_output = np.dot(this.hl_weights[i], data_i) # f(w, x) = <w, x>
            node_i = this.Leaky_ReLU(linear_output)
            output += np.dot(this.out_weights[i], node_i)

        # Output node ↓
        class_predicted = this.Sigmoid(output)
        return class_predicted

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

    def Leaky_ReLU(this, linear_output):
        return linear_output if linear_output > 0 else linear_output * 0.01
    
    def Sigmoid(this, output):
        return 1 / (1 + np.exp(-output))
    
    def ReLU_derevative(this, linear_output):
        return 1 if linear_output > 0 else 0.01
    
    def log_loss(this, class_predicted, true_class):
        y = true_class
        y_p = class_predicted
        L = -(y * np.log2(y_p) + (1 - y) * np.log2(1 - y_p))
        return L

# Useful stuff → https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b    

