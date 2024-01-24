import MLP
import dataMethods

mlp = MLP.MultiLayerPerceptron()
m_training_mae= 0
m_testing_mae = 0
m_training_mse= 0
m_testing_mse = 0


print("MultiLayerPerceptron:")
print("************")
print("CURRENTLY THIS ALGORITHM DOES NOT WORK WELL")
print("I BELIEVE WITH ENOUGH TWEAKING OF ITS MANY VARIABLES IT COULD WORK")
print("BUT THERE ARE LOT OF VARIABLES AND FAR TOO LITTLE TIME") # also it takes about 60 sec to run
print("************") 
# 10 itterations one for each fold
for i in range(10):

    mydata = dataMethods.Perceptron_Dataset(fold=i)

    # Train 💪💪💪
    mlp.Train(mydata.Training(), mydata.Training_labels())

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

print("--------- MEDIAN STATS ---------")
print("TRAINING MEAN ABSOLUTE ERROR =", m_training_mae / 10)
print("TRAINING MEAN SQUARED ERROR =", m_training_mse / 10)
print("TESTING MEAN ABSOLUTE ERROR =", m_testing_mae / 10)
print("TESTING MEAN SQUARED ERROR =", m_testing_mse / 10)
