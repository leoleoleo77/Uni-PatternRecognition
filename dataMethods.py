import data

file = 'E:/Uni/SEM5/anag prwtipwn/Ergasia/housing.csv'
mydata = data.HousingData(file).Standardize_Housing_Data()

def AllData():
    return mydata

def Longitude_column():
    longitude_data = []
    for i in range(len(mydata)):
        longitude_data.append(mydata[i][0])
    return longitude_data

def Latitude_column():
    latitude_data = []
    for i in range(len(mydata)):
        latitude_data.append(mydata[i][1])
    return latitude_data

def Housing_median_age_column():
    housing_median_age_data = []
    for i in range(len(mydata)):
        housing_median_age_data.append(mydata[i][2])
    return housing_median_age_data

def Total_rooms_column():
    total_rooms_data = []
    for i in range(len(mydata)):
        total_rooms_data.append(mydata[i][3])
    return total_rooms_data

def Total_bedrooms_column():
    total_bedrooms_data = []
    for i in range(len(mydata)):
        total_bedrooms_data.append(mydata[i][4])
    return total_bedrooms_data

def Population_column():
    population_data = []
    for i in range(len(mydata)):
        population_data.append(mydata[i][5])
    return population_data

def Households_column():
    households_data = []
    for i in range(len(mydata)):
        households_data.append(mydata[i][6])
    return households_data

def Median_income_column():
    median_income_data = []
    for i in range(len(mydata)):
        median_income_data.append(mydata[i][7])
    return median_income_data

def Median_house_value_column():
    median_house_value_data = []
    for i in range(len(mydata)):
        median_house_value_data.append(mydata[i][8])
    return median_house_value_data

def Ocean_proximity_column():
    ocean_proximity_data = []
    for i in range(len(mydata)):
        ocean_proximity_data.append(mydata[i][9])
    return ocean_proximity_data
