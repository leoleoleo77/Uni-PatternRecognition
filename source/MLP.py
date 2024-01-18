import numpy as np
import sys
sys.stdout = open('log.txt', 'w')
np.set_printoptions(threshold = np.inf)

class MLP:
    def __init__(this):
        this.lr = 0.1
        this.n_nodes = 10 # Number of nodes inside the hidden layer
        this.hl_weights = [] # hidden layer weights
        this.out_weights = None # output weights

    def Train(this, data, data_labels):
        dimensions = len(data[0]) # get the number of features/dimensions of our data
        for _ in range(this.n_nodes):
            #this.hl_weights.append(np.random.normal(0, 0.1, dimensions) * 0.1) # initialize the weights for each feature/dimention to 0
            this.hl_weights.append(np.ones(dimensions))
        #this.out_weights = np.random.normal(0, 0.1, this.n_nodes) * 0.1 # init the output weights 
        this.out_weights = np.ones(dimensions)
        for indx, data_i in enumerate(data): # for each datapoint
            Z = []
            # Hidden layer nodes ↓
            output = 0
            for i in range(this.n_nodes): # for each node inside the hidden layer
                linear_output = np.dot(this.hl_weights[i], data_i) # f(w, x) = <w, x>
                Z.append(linear_output)
                node_i = this.ReLU(linear_output)
                output += np.dot(this.out_weights[i], node_i)

            # Output node ↓
            class_predicted = this.Sigmoid(output)
            # Binary Cross-Entropy Loss ...ㄟ( ▔, ▔ )ㄏ
            loss = this.logs_loss(class_predicted, data_labels[indx])
            # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
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
    def ReLU(this, linear_output):
        return np.max([0, linear_output])
    
    def Sigmoid(this, linear_output):
        return 1 / (1 + np.exp(-linear_output))
    
    def ReLU_derevative(this, linear_output):
        return 1 if linear_output > 0 else 0
    
    def logs_loss(this, class_predicted, true_class):
        y = true_class
        y_p = class_predicted
        L = -(y * np.log2(y_p) + (1 - y) * np.log2(1 - y_p))
        return L

    

import dataMethods
p = dataMethods.Perceptron_Dataset(0)
data = p.Testing()
labels = p.Testing_labels()
mlp = MLP()
mlp.Train(data, labels)
#print(mlp.Sigmoid(-2))