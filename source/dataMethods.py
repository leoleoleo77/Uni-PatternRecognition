import math
import data
import numpy as np
import sys
import random
from pathlib import Path

# Set the default print output to be inside the log.txt file
sys.stdout = open('log.txt', 'w')
# np.set_printoptions(threshold = np.inf)

# Get the relative path of the housing.csv file
p = Path(__file__).with_name('housing.csv')
file_path = p.absolute()

# Create an instance of the HousingData class
mydata = data.HousingData(file_path).Standardize_Housing_Data()

# Calculate the number of rows (20640)
num_of_data = len(mydata)

# Shuffle the data
random.shuffle(mydata)


def AllData():
    return mydata

# Get the column of each feature
def Longitude_column():
    longitude_data = []
    for i in range(num_of_data):
        longitude_data.append(mydata[i][0])
    return longitude_data

def Latitude_column():
    latitude_data = []
    for i in range(num_of_data):
        latitude_data.append(mydata[i][1])
    return latitude_data

def Housing_median_age_column():
    housing_median_age_data = []
    for i in range(num_of_data):
        housing_median_age_data.append(mydata[i][2])
    return housing_median_age_data

def Total_rooms_column():
    total_rooms_data = []
    for i in range(num_of_data):
        total_rooms_data.append(mydata[i][3])
    return total_rooms_data

def Total_bedrooms_column():
    total_bedrooms_data = []
    for i in range(num_of_data):
        total_bedrooms_data.append(mydata[i][4])
    return total_bedrooms_data

def Population_column():
    population_data = []
    for i in range(num_of_data):
        population_data.append(mydata[i][5])
    return population_data

def Households_column():
    households_data = []
    for i in range(num_of_data):
        households_data.append(mydata[i][6])
    return households_data

def Median_income_column():
    median_income_data = []
    for i in range(num_of_data):
        median_income_data.append(mydata[i][7])
    return median_income_data

def Median_house_value_column():
    median_house_value_data = []
    for i in range(num_of_data):
        median_house_value_data.append(mydata[i][8])
    return median_house_value_data

def Ocean_proximity_column():
    ocean_proximity_data = []
    for i in range(num_of_data):
        ocean_proximity_data.append(mydata[i][9])
    return ocean_proximity_data

# All the data but without the median house value
def No_Median_house_value():
    no_median_house_value_data = mydata[:] # is like passing mydata by value
    for i in range(num_of_data):
        no_median_house_value_data[i][8] = no_median_house_value_data[i][9]
        no_median_house_value_data[i].pop()
    return no_median_house_value_data

class Perceptron_Dataset:
    labels = []
    def __init__(this, fold, threshold=0.34):
        this.threshold = threshold
        this.sample_size = math.ceil(num_of_data * 0.1)
        this.training_data = []
        this.training_labels = []
        this.testing_data = []
        this.testing_labels = []
        #for this.folds in range(1):
        this.folds = fold
        if fold == 0: this.Create_labels()
        this.Resampling_Procedure()

    def Create_labels(this):
        for i in range(num_of_data):
            label = 1.0 if mydata[i].pop(8) < this.threshold else 0.0
            Perceptron_Dataset.labels.append(label)

    def Resampling_Procedure(this):
        fold_correct_index = this.sample_size * this.folds
        k = fold_correct_index
        for i in range(num_of_data):
            if i < this.sample_size:
                this.testing_labels.append(Perceptron_Dataset.labels[k])
                this.testing_data.append(mydata[k][:-1]) # Save the training data 
                for x_j in mydata[k][-1]:
                    this.testing_data[i].append(float(x_j)) 
                this.testing_data[i].insert(0, 1.0) # add the weight of the bias at the start of the list
            else:
                this.training_labels.append(Perceptron_Dataset.labels[k])
                this.training_data.append(mydata[k][:-1])
                for x_j in mydata[k][-1]:
                    this.training_data[i - this.sample_size].append(float(x_j)) 
                this.training_data[i - this.sample_size].insert(0, 1.0)
            k = k + 1 if k < num_of_data - 1 else 0

    def Training(this):
        return np.array(this.training_data, dtype=object)
    def Training_labels(this):
        return np.array(this.training_labels, dtype=object)
    def Testing(this):
        return np.array(this.testing_data, dtype=object)
    def Testing_labels(this):
        return np.array(this.testing_labels, dtype=object)

    