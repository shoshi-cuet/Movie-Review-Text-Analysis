import csv
import json
import os

data_directory = r"C:\Users\Rimel\Desktop\DAT 500 Final Project\Dataset\Original Data"
movie_data = []
for filename in os.listdir(data_directory):
    if filename.endswith(".json"):
        with open(f'{data_directory}/{filename}') as json_file:
            data = json.load(json_file)
            movie_data.append(data)

data = []
for m in movie_data:
    for item in m:
        data.append(item)

# now we will open a file for writing
test_file = open(r'C:/Users/Rimel/Desktop/DAT 500 Final Project/Dataset/Combined Data/combined_data.csv', 'w', encoding='utf-8')
 
# create the csv writer object
csv_writer = csv.writer(test_file, lineterminator='\n')
 
# Counter variable used for writing
# headers to the CSV file
count = 0
 
for row in data:
    if count == 0:
        header = row.keys()
        csv_writer.writerow(header)
        count += 1

    csv_writer.writerow(row.values())
 
test_file.close()
