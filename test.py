import csv

longitudeData = []
with open('E:/Uni/SEM5/anag prwtipwn/housing.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    print(list(spamreader)[0][0])
