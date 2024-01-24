import perceptron
import dataMethods

p = perceptron.Perceptron()
m_training_mae= 0
m_testing_mae = 0

print("PERCEPTRON:")
print("************")
print("ALL STATS ARE REPRESENTATIVE OF THE MEAN ABSOLUTE ERROR")
print("AND SINCE THIS IS A BINARY DISTRIBUTION PROBLEM")
print("THEY ARE ALSO REPRESENTATIVE OF THE MEAN SQUARED ERROR")
print("************")
# 10 itterations one for each fold
for i in range(10):

    mydata = dataMethods.Perceptron_Dataset(fold=i)

    # Train 💪💪💪
    p.Train_weights(mydata.Training(), mydata.Training_labels())

    if i == 0: print("CLASS DISTRIBUTION RATIO:", (sum(mydata.labels)/dataMethods.num_of_data))
    print("--------- fold -", i, "---------")

    # Training stats
    training_mae = p.Mean_absolute_error(mydata.Training(), mydata.Training_labels())
    print("TRAINING ERROR =", training_mae)
    # Testing stats
    testing_mae = p.Mean_absolute_error(mydata.Testing(), mydata.Testing_labels())
    print("TESTING ERROR =", testing_mae)
    
    # sum errors to find the median error after the loop
    m_training_mae += training_mae
    m_testing_mae += testing_mae

print("--------- MEDIAN STATS ---------")
print("MEDIAN TRAINING ERROR =", m_training_mae / 10)
print("MEDIAN TESTING ERROR =", m_testing_mae / 10)
