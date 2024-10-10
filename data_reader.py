import numpy as np
import os


def get_mri_data(subject, data_folder):
    """Reads MRI data for the specified subject.

    Args:
        subject (str): Subject ID
        data_folder (str): Absolute path to the data folder.
    """
    file_path = os.path.join(data_folder, f"sub-{subject}")
    return
