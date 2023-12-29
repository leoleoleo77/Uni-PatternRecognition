
'''
Προ-επεξεργασία Δεδομένων

1.	Θα πρέπει να αναγνωρίσετε τα υποσύνολα των αριθμητικών και των κατηγορικών χαρακτηριστικών.
2.	Για το υποσύνολο των αριθμητικών χαρακτηριστικών θα πρέπει να πειραματιστείτε με διαφορετικές τεχνικές 
    κλιμάκωσης (scaling) των δεδομένων ώστε όλα τα αριθμητικά χαρακτηριστικά να αναπαρίστανται στην ίδια κλίμακα.
3.	Για το υποσύνολο των κατηγορικών χαρακτηριστικών μπορείτε να χρησιμοποιήσετε την One Hot Vector 
    κωδικοποίηση ώστε τα εν λόγω δεδομένα να λάβουν διανυσματική αναπαράσταση.
4.	Θα πρέπει να αναγνωρίσετε αν υπάρχουν αριθμητικά χαρακτηριστικά με ελλιπείς τιμές.
    Για τις συγκεκριμένες εγγραφές μπορείτε να συμπληρώσετε τις τιμές που απουσιάζουν με την διάμεση τιμή του χαρακτηριστικού.
'''

import csv
from sklearn.preprocessing import MinMaxScaler

class HousingData:
    def __init__(this, csv_file_location):
        this.csv_file_location = csv_file_location

    def Standardize_Housing_Data(this):
        # ocean_proximity options: ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
        # one hot vector 
        def Standardize_ocean_proximit(ocean_proximity):
            if ocean_proximity == "NEAR BAY":
                return [1, 0, 0, 0 ,0]
            elif ocean_proximity == "<1H OCEAN":
                return [0, 1, 0, 0 ,0]
            elif ocean_proximity == "INLAND":
                return [0, 0, 1, 0 ,0]
            elif ocean_proximity == "NEAR OCEAN":
                return [0, 0, 0 ,1 ,0]
            else:
                return [0, 0, 0 ,0 ,1]

        file = open(this.csv_file_location) # 'E:/Uni/SEM5/anag prwtipwn/housing.csv'
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)

        # The rows[] list where we append all the data
        rows = []

        # The total_bedrooms column contains some empty cells
        # We find the median value of all the non-empty cells and fill the empty cells with that value

        # total_bedrooms_sum is the sum of all the non-empty cells
        total_bedrooms_sum = 0

        # total_bedrooms_empty[] contains the index of each row that lacks data for the number of total_bedrooms
        total_bedrooms_emptyIndexes = [] 

        # index of the row
        i = 0

        # For each row inside the housing.csv file
        for row in csvreader:
            # For each cell inside a row
            for data in row:
                index = row.index(data) # index of the cell
                if index != 9:
                    if index != 4: 
                        data = float(data)
                    else:
                        if data == "":
                            total_bedrooms_emptyIndexes.append(i)
                        else:
                            data = float(data)
                            total_bedrooms_sum += data
                else:
                    data = Standardize_ocean_proximit(data)
                row[index] = data
            rows.append(row)
            i += 1

        # The median value of all the non-empty cells
        total_bedrooms_medianValue = total_bedrooms_sum / len(rows) - len(total_bedrooms_emptyIndexes)

        # fill the empty cells in the total_bedrooms column with the median value
        for empty_data_index in total_bedrooms_emptyIndexes:
            rows[empty_data_index][4] = total_bedrooms_medianValue

        # remove the ocean_proximity column to scale the features
        # temp_rows is the temporary list of the rows but without the ocean_proximity column
        temp_rows=[]
        # ocean_proximity_list is the list of the ocean_proximity column data, so we can add it back later to the rows list
        ocean_proximity_list = []
        for row in rows:
            ocean_proximity_list.append(row.pop())
            temp_rows.append(row)

        # scale features
        scaler = MinMaxScaler()
        model=scaler.fit(temp_rows)
        scaled_data=model.transform(temp_rows)

        # add the ocean_proximity column data back in the rows list
        for i in range(len(rows)):
            rows[i] = list(scaled_data[i])
            rows[i].append(ocean_proximity_list[i])
        # print scaled features
        file.close()
        return rows
