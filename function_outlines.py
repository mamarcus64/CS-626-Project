import numpy as np
from numpy.typing import NDArray
from typing import Dict


def get_llm_vector_representations(input_sentence: str, model_name: str) -> np.array:
    """Calculates the layer-wise LLM vector embeddings of the input sentence using the specified model.

    Args:
        input_sentence (str): Input sentence to get vector embeddings from
        model_name (str): Name of the Huggingface model to download and use

    Returns:
        np.array: An L X E numpy array, where:
            L is the number of attention layers in the model (i.e., LLaMA-7B has 32 layers)
            E is the embedding dimension of the model's layer vectors (i.e., for LLaMA-7B, 4096)
    """
    pass

def get_mri_voxel_representations(subject: str, event: Dict, regions: np.array) -> np.array:
    """Returns the fMRI scores for the specified voxels and event.

    Args:
        subject (str): Name of the subject.
        event (Dict): All of the relevant metadata to determine:
            - the location of the MRI file
            - the specific event, starting time, and sentence of the event
            - whatever else might be useful
        regions (np.array): The specific voxels to look at 
        (maybe we want to have like ~3-4 baseline regions to start off with?
        whatever you think would be best idk how this works)

    Returns:
        np.array: Returns a 1-dimensional vector where each cell corresponds to one MRI voxel.
        The MRI 3D regions are flattened into a one-dimensional vector. I'm not sure how to average the scores over time,
        but we need to do that to make the output 1-dimensional. Maybe we can experiment with different timings/averages.
    """
    pass

def create_llm_and_mri_vectors(input_sentence, subjects):
    # Emily make doc string
    pass

def calculate_brain_score(population):
    # Emily make doc string
    pass

def get_populations(population_index):
    # Nona make docstring
    # Hand-create ~5-6 populations that would cover potentially interesting findings and effects. 
    # This function just returns one of these populations based on the index
    pass

# add some functions for statistical tests, not sure what they would be