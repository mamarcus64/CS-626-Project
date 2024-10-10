import numpy as np
import data_reader as dr

from numpy.typing import NDArray
from scipy.stats import pearsonr
from typing import Dict


DATA_FOLDER = ""
MODEL_NAME = ""


def get_llm_vector_repesentations(input_sentence: str, model_name: str) -> np.array:
    """Calculates the layer-wise LLM vector embeddings of the input sentence using the specified model.

    Args:
        input_sentence (str): Input sentence to get vector embeddings from
        model_name (str): Name of the Huggingface model to download and use

    Returns:
        np.array: An L X E numpy array, where:
            L is the number of attention layers in the model (i.e., LLaMA-7B has 32 layers)
            E is the embedding dimension of the model's layer vectors (i.e., for LLaMA-7B, 4096)
    """
    # placeholder values
    return np.random.rand((10, 768))

def get_mri_voxel_repesentations(subject: str, regions: np.array) -> np.array:
    """Returns the fMRI data for the specified voxels and event.

    Args:
        subject (str): Subject ID.
        regions (np.array): The specific voxels to look at 
        (maybe we want to have like ~3-4 baseline regions to start off with?
        whatever you think would be best idk how this works)

    Returns:
        np.array: A 1 x n numpy array/1-dimensional vector where each cell corresponds to one MRI voxel.
        The MRI 3D regions are flattened into a 1-dimensional vector. I'm not sure how to average the scores over time,
        but we need to do that to make the output 1-dimensional. Maybe we can experiment with different timings/averages.
    """
    data = dr.get_mri_data(subject, DATA_FOLDER)
    # placeholder values
    data = np.random.rand((1, 500))
    return data

def create_llm_and_mri_vectors(input_sentence, subjects, regions):
    # TODO: Focus on one region at a time?
    """Returns layer-wise LLM vector embeddings of the input sentence and the corresponding fMRI repesentations from the subjects.
    
    Args: 
        input_sentence (str): Input sentence to get vector embeddings from
        subjects (list): Subjects to get fMRI data from 
        model_name (str): Name of the Huggingface model to download and use
        regions (np.array): The specific voxels to look at 

    Returns: 
        np.array: An L X E numpy array, where:
            L is the number of attention layers in the model (i.e., LLaMA-7B has 32 layers)
            E is the embedding dimension of the model's layer vectors (i.e., for LLaMA-7B, 4096)
        np.array: An S x n numpy array, where:
            S is the number of subjects
            n is the number of voxels in the fMRI data
    """
    llm_rep = get_llm_vector_repesentations(input_sentence, MODEL_NAME)
    mri_rep = []
    for subject in subjects:
        mri_data = get_mri_voxel_repesentations(subject, regions)
        mri_rep.append(mri_data)
    mri_rep = np.hstack(mri_rep)

    return llm_rep, mri_rep

def calculate_brain_score(model, subjects, input_sentences, regions):
    # We should focus on one layer of the LLM at a time
    """Returns an array of brain scores for a given set of sentences. 
    Retrieves the LLM and fMRI repesentations of the sentences for each subject and 

    Args: 
        model (sklearn.linear_model.LinearRegression): Trained linear regression model.
        subjects (list): Subjects to get fMRI data from 
        input_sentences (np.array): Input sentences for linear regression model.

    Returns:
        np.array: A 1 x S numpy array, where each element corresponds to the average Pearson correlation coefficient between
        the predicted and actual neural data for each sentence.
        # TODO: Should we average r over all subjects for a given sentence? 
    """
    x = []
    y = []
    for sentence in input_sentences:
        llm_rep, mri_rep = create_llm_and_mri_vectors(sentence, subjects, regions)
        # llm_rep: (L x E) --> We should have L = 1 in order to have consistent brain scores
        # mri_rep: (S x n)
        x.append(llm_rep)
        y.append(mri_rep)
    x = np.hstack(x)
    y = np.hstack(y)

    neural_predictions = model.predict(x)
    r = pearsonr(neural_predictions, y)    # Check shape of correlation coefficients -- should have one per sentence
    return r


def get_populations(population_index):
    # Nona make docstring
    # Hand-create ~5-6 populations that would cover potentially interesting findings and effects. 
    # This function just returns one of these populations based on the index
    pass

# add some functions for statistical tests, not sure what they would be