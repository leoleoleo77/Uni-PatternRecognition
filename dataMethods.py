import math
import data

file = 'C:/Users/User/Desktop/Uni/sem5/anag prwtypwn/Uni-PatternRecognition-main/housing.csv'
mydata = data.HousingData(file).Standardize_Housing_Data()
num_of_data = len(mydata)

def AllData():
    return mydata

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

def Training_data(percentage):
    num_of_training_data = math.ceil(num_of_data * percentage)
    training_data = No_Median_house_value()
    return num_of_training_data

# TODO
class Perceptron_data:
    def __init__(this, portion=0.9):
        this.num_of_training_data = math.ceil(num_of_data * portion)
        this.num_of_testing_data = num_of_data - this.num_of_training_data
    def Training(this):

        return this.num_of_testing_data
    def Training_labels(): pass
    def Testing(): pass
    def Testing_labels(): pass

p = Perceptron_data()
p.Training()
    