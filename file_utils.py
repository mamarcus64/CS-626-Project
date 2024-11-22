import os
import sys

PARENT_PATH = os.path.abspath(os.path.dirname(__file__))
BRAIN_SCORE_LANGUAGE_REPO_PATH = os.path.join(PARENT_PATH, 'brain-score-language')
NEURAL_COSINE_SAVE_PATH = os.path.join(PARENT_PATH, 'neural_cosine')
IDIOMS_DATA_PROCESSING_PATH = os.path.join(PARENT_PATH, 'idioms_data_processing')

# Append local brain-score repo to path, brittle but should be fine
sys.path.append(BRAIN_SCORE_LANGUAGE_REPO_PATH)

"""
# Include the following lines at the top of each file for file pathing.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../')) # adjust relative path as needed
import file_utils
"""