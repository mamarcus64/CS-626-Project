import csv
import os

duration = 2
height = 1
run_num = str(3)
# Define the path for the input CSV file
csv_file_path = "/Users/helen/Desktop/GI/GroupA/Stimuli_GroupA_run" + run_num + "_Sorted.csv"

# Define the folder where the .txt files will be stored
output_folder = "/Users/helen/Desktop/GI/GroupA/EV_files/run" + run_num
os.makedirs(output_folder, exist_ok=True)

output_file_path_hashtag = os.path.join(output_folder, f'EV_71.txt')

hashtag_rows = []

# Read each line from the CSV file and store it in a separate .txt file
with open(csv_file_path, mode='r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:

        onset = int(row[0])
        onset = round(onset/1000, 2)

        condition = row[2]
        sentence_code = int(row[3])

        content = str(onset) + "\t" + str(duration) + "\t" + str(height) + "\n"
        print(sentence_code, condition, content)

        if condition != "77":

            EV_code = row[4]

            if int(EV_code) < 10:
                EV_code = "0"+EV_code

            output_file_path = os.path.join(output_folder, f'EV_{EV_code}.txt')
            
            # Write into the .txt file
            with open(output_file_path, mode='w') as output_file:
                output_file.write(content)  

        else:
            
            hashtag_rows.append(content)

hashtag_rows_sorted = sorted(hashtag_rows, key=lambda x: float(x.split('\t')[0]))

with open(output_file_path_hashtag, mode='w') as output_file:
    for item in hashtag_rows_sorted:
        output_file.write(item)  
















