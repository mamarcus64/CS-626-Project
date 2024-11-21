
# To Skull-strip T1 -----------------------
import subprocess
import time

# Define how many times you want to run the script, or run indefinitely
num_subj = 23  # Set this to your desired number of runs
delay = 3  # Set a delay (in seconds) between runs
target_script = "/Users/helenwu/Desktop/German_idioms/skull_strip_byfile.py"

for i in range(21,22):

    subj_i = i+1

    if subj_i < 10:
        subj_n = "0" + str(subj_i)
    else:
        subj_n = str(subj_i)

    input_file = "/Users/helenwu/Desktop/German_idioms/ds002727/sub-" + subj_n + "/anat/sub-" + subj_n + "_T1w.nii.gz"
    output_file = "/Users/helenwu/Desktop/German_idioms/ds002727/sub-" + subj_n + "/anat/sub-" + subj_n + "_T1w_brain.nii.gz"

    print()
    print(subj_n)
    print(input_file)
    print(output_file)

    # Run the target script with additional arguments
    subprocess.run(["python", target_script, "--infile", input_file, "--outfile", output_file])
    # Optional delay between runs
    time.sleep(delay)




# # To Skull-strip Mag -----------------------
# import subprocess
# import time

# # Define how many times you want to run the script, or run indefinitely
# num_subj = 23  # Set this to your desired number of runs
# delay = 3  # Set a delay (in seconds) between runs
# target_script = "/Users/helenwu/Desktop/German_idioms/mm_skull_strip_byfile.py"

# for i in range(num_subj):

#     subj_i = i+1

#     if subj_i < 10:
#         subj_n = "0" + str(subj_i)
#     else:
#         subj_n = str(subj_i)

#     input_file = "/Users/helenwu/Desktop/German_idioms/ds002727/sub-" + subj_n + "/fmap/sub-" + subj_n + "_magnitude1.nii.gz"
#     output_file = "/Users/helenwu/Desktop/German_idioms/ds002727/sub-" + subj_n + "/fmap/sub-" + subj_n + "_magnitude1_brain.nii.gz"

#     print()
#     print(subj_n)  
#     print(input_file)
#     print(output_file)

#     # Run the target script with additional arguments
#     subprocess.run(["python", target_script, "--infile", input_file, "--outfile", output_file])
#     # Optional delay between runs
#     time.sleep(delay)











