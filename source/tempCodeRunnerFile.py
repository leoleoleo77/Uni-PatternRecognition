mydata = dataMethods.Perceptron_Dataset(fold=i)

    # Train 💪💪💪
    p.Train_weights(mydata.Training(), mydata.Training_labels())