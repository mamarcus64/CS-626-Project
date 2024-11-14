import numpy as np
import os

def generate_dummy_data(num_subjects, voxels_per_subject, num_stimuli):
    return np.random.rand(num_stimuli, num_subjects * voxels_per_subject)

data = generate_dummy_data(10, 15, 12)
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "processed_data"))
file_path = os.path.join(parent_directory, "dummy_data.npy")
np.save(file_path, data)