# Include the following lines at the top of each file for file pathing.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # adjust relative path as needed
import file_utils

import numpy as np
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from brainscore_core.metrics import Score
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

from brainscore_language.benchmarks.german_emotive_idioms import GermanEmotiveIdioms

def brainscore(model_name: str,
               layer_name: str,
               neural_data: np.ndarray,
               stimuli_ids: List,
               ceiling: Score, 
               ood_testing_data: Optional[List[Tuple[np.ndarray, List]]] = None,
               data_names: Optional[List[str]] = None,
               model: Optional[AutoModelForCausalLM] = None,
    ) -> dict:
    """Calculates brain score and returns a formatted dictionary containing the overall brain scores and
    neuron-specific brain scores for all data provided. Regreesions are trained on neural_data,
    and ood_testing_data is tested on the regressions of neural_data. For now, assumes one subject.

    Args:
        model_name (str): name of Huggingface model
        layer_name (str): name of specific model layer to evaluate
        neural_data (np.ndarray): array of shape (stimuli, voxels_per_subject)
        stimuli_ids (List): list of stimuli ids for each row in neural_data
        ceiling (Score): ceiling value for the neural_data
        ood_testing_data (Optional[List[Tuple[np.ndarray, List]]], optional): list of tuples of ood_neural_data, ood_stimuli_ids.
        data_names (Optional[List[str]], optional): names for all of the data tested (including neural_data). Defaults to None.
        model (Optional[AutoModelForCausalLM]): transformers model. Recommended to directly pass in if calling this function many times.
    """
    
    # convert ood_testing_data to list
    if ood_testing_data is None:
        ood_testing_data = []
    if type(ood_testing_data) == np.ndarray:
        ood_testing_data = [ood_testing_data]
        
    # convert data names, if none passed in
    if data_names is None:
        data_names = [f'data_{i}' for i in range(1 + len(ood_testing_data))] # the + 1 corresponds to data name for 'neural_data'
    
    assert len(data_names) == 1 + len(ood_testing_data), 'Length of "data_names" does not match with number of data provided'
    
    num_subjects = 1 # for now, we are assuming only one subject. this could change in the future
    
    benchmark = GermanEmotiveIdioms(
        neural_data = neural_data,
        num_subjects = num_subjects,
        selected_stimuli_ids = stimuli_ids,
        ceiling = ceiling,
        metric = None, # will default to linear_pearsonr_unaveraged
    )
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
    
    hf_subject = HuggingfaceSubject(
        model_id=model_name,
        model=model,
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_name}
    )
    
    brainscore_results = {
        'brain_score': {},
        'neuron_values': {}
    }
    
    # get results for neural_data (with training regression)
    scores = benchmark(candidate=hf_subject, train_regression=True)
    brainscore_results['brain_score'][data_names[0]] = float(np.median(scores))
    brainscore_results['neuron_values'][data_names[0]] = [float(x) for x in scores]
    
    # get results for ood_testing_data (without training regression)
    for data_name, data in zip(data_names[1:], ood_testing_data):
        # load new data into benchmark
        ood_neural_data, ood_stimuli_ids = data
        benchmark.load_data_into_assembly(
            neural_data=ood_neural_data,
            num_subjects=num_subjects,
            selected_stimuli_ids=ood_stimuli_ids
        )
        
        # calculate ood brain scores
        scores = benchmark(candidate=hf_subject, train_regression=False)
        brainscore_results['brain_score'][data_name] = float(np.median(scores))
        brainscore_results['neuron_values'][data_name] = [float(x) for x in scores]
        
    return brainscore_results