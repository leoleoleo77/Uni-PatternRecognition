import MLP
import dataMethods

mlp = MLP.MultiLayerPerceptron()
m_training_mae= 0
m_testing_mae = 0
m_training_mse= 0
m_testing_mse = 0


print("MultiLayerPerceptron:")
print("************")
print(" X~~~X ")
print("************")
# 10 itterations one for each fold
for i in range(2):

    #mydata = dataMethods.Perceptron_Dataset(fold=i)
    mydata = dataMethods.Perceptron_Dataset(fold=i)

    # Train 💪💪💪
    mlp.Train(mydata.Training()[:-10000], mydata.Training_labels()[:-10000])

    if i == 0: print("CLASS DISTRIBUTION RATIO:", (sum(mydata.labels)/dataMethods.num_of_data))
    print("--------- fold -", i, "---------")

    # Training stats
    training_mae = mlp.Mean_absolute_error(mydata.Training(), mydata.Training_labels())
    training_mse = mlp.Mean_squared_error(mydata.Training(), mydata.Training_labels())
    print("TRAINING MEAN ABSOLUTE ERROR =", training_mae)
    print("TRAINING MEAN SQUARED ERROR =", training_mse)

    # Testing stats
    testing_mae = mlp.Mean_absolute_error(mydata.Testing(), mydata.Testing_labels())
    testing_mse = mlp.Mean_squared_error(mydata.Testing(), mydata.Testing_labels())
    print("TESTING MEAN ABSOLUTE ERROR =", testing_mae)
    print("TESTING MEAN SQUARED ERROR =", testing_mse)

    # sum errors to find the median error after the loop
    m_training_mae += training_mae
    m_testing_mae += testing_mae

    m_training_mse += training_mse
    m_testing_mse += testing_mse

'''print("--------- MEDIAN STATS ---------")
print("TRAINING MEAN ABSOLUTE ERROR =", training_mae / 10)
print("TRAINING MEAN SQUARED ERROR =", training_mse / 10)
print("TESTING MEAN ABSOLUTE ERROR =", testing_mae / 10)
print("TESTING MEAN SQUARED ERROR =", testing_mse / 10)'''
