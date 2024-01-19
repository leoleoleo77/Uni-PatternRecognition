import linear_regression as linreg
import dataMethods

LSM = linreg.Least_Squares_Method()
m_training_mae= 0
m_testing_mae = 0
m_training_mse= 0
m_testing_mse = 0


print("LEAST SQUARES METHOD:")
print("************")
print(" <(_  _)>~~? ")
print("************")
# 10 itterations one for each fold
for i in range(10):

    mydata = dataMethods.Linear_Regression_Dataset(fold=i)

    # Train 💪💪💪
    LSM.Train(mydata.Training(), mydata.Training_labels())

    print("--------- fold -", i, "---------")

    # Training stats
    training_mae = LSM.Mean_absolute_error(mydata.Training(), mydata.Training_labels())
    training_mse = LSM.Mean_squared_error(mydata.Training(), mydata.Training_labels())
    print("TRAINING MEAN ABSOLUTE ERROR =", training_mae)
    print("TRAINING MEAN SQUARED ERROR =", training_mse)

    # Testing stats
    testing_mae = LSM.Mean_absolute_error(mydata.Testing(), mydata.Testing_labels())
    testing_mse = LSM.Mean_squared_error(mydata.Testing(), mydata.Testing_labels())
    print("TESTING MEAN ABSOLUTE ERROR =", testing_mae)
    print("TESTING MEAN SQUARED ERROR =", testing_mse)
    
    # sum errors to find the median error after the loop
    m_training_mae += training_mae
    m_testing_mae += testing_mae

    m_training_mse += training_mse
    m_testing_mse += testing_mse

print("--------- MEDIAN STATS ---------")
print("TRAINING MEAN ABSOLUTE ERROR =", training_mae / 10)
print("TRAINING MEAN SQUARED ERROR =", training_mse / 10)
print("TESTING MEAN ABSOLUTE ERROR =", testing_mae / 10)
print("TESTING MEAN SQUARED ERROR =", testing_mse / 10)
