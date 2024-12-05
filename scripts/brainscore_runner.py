# Include the following lines at the top of each file for file pathing.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # adjust relative path as needed
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import file_utils

import h5py
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import brainscore_calculation # from scripts folder
from brainscore_core.metrics import Score

def load_neural_data(h5_filename, standardize=True):
    neural_data = None
    with h5py.File(os.path.join(file_utils.VOXEL_PATH, h5_filename), 'r') as f:
        neural_data = f['fMRI_sentences'][:]
        
    standardize = True
    if standardize:  # mean=0, std=1
        mean = np.mean(neural_data)
        std = np.std(neural_data)
        neural_data = (neural_data - mean) / std
        
    transposed_neural_data = np.transpose(neural_data, axes=(2, 1, 0))
    reshaped_neural_data = transposed_neural_data.reshape(neural_data.shape[2], -1)
    return reshaped_neural_data

def load_ceiling(ceiling_filename):
    ceilings_file = os.path.join(file_utils.CEILING_PATH, ceiling_filename)
    with open(ceilings_file, "r") as f:
        ceiling = json.load(f)

    # Cast ceiling to Score
    score = ceiling['score']
    raw = ceiling['raw']
    ceiling = Score(score)
    ceiling.raw = raw
    ceiling.name = 'data'
    return ceiling

H5_FILE = 'fMRI_four_regions_ordered_data.h5'
CEILING_FILE = 'fMRI_semantic_all_language_areas_ceilings.json' 
NEURAL_DATA = load_neural_data(H5_FILE)
CEILING = load_ceiling(CEILING_FILE)

def get_model_layers(model_name):
    if model_name == 'distilgpt2':
        layer_names = [f'transformer.h.{block}.ln_1' for block in range(6)]
    elif model_name == 'TUM/GottBERT_base_last':
        layer_names = [f'roberta.encoder.layer.{block}.output.dense' for block in range(12)]
    elif model_name == 'DiscoResearch/Llama3-German-8B':
        layer_names = [f'model.layers.{block}.input_layernorm' for block in range(32)]
    elif model_name == 'meta-llama/Llama-3.2-1B':
        layer_names = [f'model.layers.{block}.input_layernorm' for block in range(16)]
    return layer_names

def get_middle_layer(model_name):
    model_layers = get_model_layers(model_name)
    if model_name == 'distilgpt2':
        return model_layers[3]
    elif model_name == 'TUM/GottBERT_base_last':
        return model_layers[6]
    elif model_name == 'DiscoResearch/Llama3-German-8B':
        return model_layers[16]
    elif model_name == 'meta-llama/Llama-3.2-1B':
        return model_layers[8]
    
def load_condition(neural_data, condition, subject, voxels_per_subject=1785):
    index_start = subject * voxels_per_subject
    index_end = (subject + 1) * voxels_per_subject
    
    if condition == 'all':
        data = neural_data[:, index_start:index_end]
        stimuli = list(range(1, 181))
    if condition == 'idiom':
        data = neural_data[0:90, index_start:index_end]
        stimuli = list(range(1, 91))
    if condition == 'literal':
        data = neural_data[90:180, index_start:index_end]
        stimuli = list(range(91, 181))
    if condition == 'negative':
        negative_idioms = neural_data[0:30, index_start:index_end]
        negative_literals = neural_data[90:120, index_start:index_end]
        data = np.concatenate((negative_idioms, negative_literals), axis=0)
        stimuli = list(range(1, 31)) + list(range(91, 121))
    if condition == 'neutral':
        neutral_idioms = neural_data[30:60, index_start:index_end]
        neutral_literals = neural_data[120:150, index_start:index_end]
        data = np.concatenate((neutral_idioms, neutral_literals), axis=0)
        stimuli = list(range(31, 61)) + list(range(121, 151))
    if condition == 'positive':
        positive_idioms = neural_data[60:90, index_start:index_end]
        positive_literals = neural_data[150:180, index_start:index_end]
        data = np.concatenate((positive_idioms, positive_literals), axis=0)
        stimuli = list(range(61, 91)) + list(range(151, 181))
        
    return data, stimuli

def run_and_save(
    model_name,
    layer_name,
    condition,
    model=None,
    save_folder='results',
    num_subjects=22,
):
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
    for subject in tqdm(list(range(num_subjects))):
        base_data, base_stimuli = load_condition(NEURAL_DATA, condition, subject)
        if condition == 'all':
            ood_testing_data = []
            data_names = ['all']
        elif condition == 'literal':
            ood_testing_data = [load_condition(NEURAL_DATA, 'idiom', subject)]
            data_names = ['literal', 'idiom']
        elif condition == 'idiom':
            ood_testing_data = [load_condition(NEURAL_DATA, 'literal', subject)]
            data_names = ['idiom', 'literal']
        elif condition == 'negative':
            ood_testing_data = [load_condition(NEURAL_DATA, 'neutral', subject), load_condition(NEURAL_DATA, 'positive', subject)]
            data_names = ['negative', 'neutral', 'positive']
        elif condition == 'neutral':
            ood_testing_data = [load_condition(NEURAL_DATA, 'negative', subject), load_condition(NEURAL_DATA, 'positive', subject)]
            data_names = ['neutral', 'negative', 'positive']
        elif condition == 'positive':
            ood_testing_data = [load_condition(NEURAL_DATA, 'negative', subject), load_condition(NEURAL_DATA, 'neutral', subject)]
            data_names = ['positive', 'negative', 'neutral']
    
        score = brainscore_calculation.brainscore(
            model_name=model_name,
            layer_name=layer_name,
            neural_data=base_data,
            stimuli_ids=base_stimuli,
            ceiling=CEILING,
            ood_testing_data=ood_testing_data,
            data_names=data_names,
            model=model,
        )
    
        # append metadata
        score['ceiling'] = CEILING.item()
        score['layer'] = layer_name
        score['model_name'] = model_name
        score['data_file'] = H5_FILE
        score['subject'] = subject
        
        # round floats to use less storage data
        for datum in score['neuron_values'].keys():
            score['neuron_values'][datum] = [round(x, 5) for x in score['neuron_values'][datum]]
        
        save_path = os.path.join(
            file_utils.PARENT_PATH,
            save_folder,
            model_name.replace('/', '_'),
            condition,
            layer_name,
        )
        
        os.makedirs(save_path, exist_ok=True)
        json.dump(score, open(os.path.join(save_path, f'{str(subject)}.json'), 'w'), indent=4)

def get_models():
    return [
        'distilgpt2',
        'TUM/GottBERT_base_last',
        'meta-llama/Llama-3.2-1B',
        'DiscoResearch/Llama3-German-8B'
    ]

def layer_analysis():
    model_name = 'meta-llama/Llama-3.2-1B'
    layer_names = [
        'model.layers.8.self_attn.q_proj',
        'model.layers.8.self_attn.k_proj',
        'model.layers.8.self_attn.v_proj',
        'model.layers.8.self_attn.o_proj',
        'model.layers.8.mlp',
        'model.layers.8.mlp.gate_proj',
        'model.layers.8.mlp.up_proj',
        'model.layers.8.mlp.down_proj',
        'model.layers.8.input_layernorm',
        'model.layers.8.post_attention_layernorm',
    ]

    neural_data = load_neural_data(H5_FILE)
    ceiling = load_ceiling(CEILING_FILE) 

    subject_num = 5

    base_data = neural_data[:90, subject_num * 1785:(subject_num + 1) * 1785]
    base_stimuli = list(range(1, 91))

    ood_data = neural_data[90:, subject_num * 1785:(subject_num + 1) * 1785]
    ood_stimuli = list(range(91, 181))

    scores = []

    model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)

    for layer_name in tqdm(layer_names):
        score = brainscore_calculation.brainscore(
            model_name=model_name,
            layer_name=layer_name,
            neural_data=base_data,
            stimuli_ids=base_stimuli,
            ceiling=ceiling,
            ood_testing_data=[(ood_data, ood_stimuli)],
            data_names=['idiomatic', 'literal'],
            model=model,
        )
        
        # append metadata
        score['ceiling'] = ceiling.item()
        score['layer'] = layer_name
        score['model_name'] = model_name
        score['data_file'] = H5_FILE
        score['subject_num'] = subject_num
        
        
        scores.append(score)
        print(layer_name)
        print(score['brain_score'])
        
    for score, layer_name in zip(scores, layer_names):
        print(layer_name)
        print(score['brain_score'])
 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for brain score.")
    
    parser.add_argument(
        "--model_name",
        type=str,
        choices=get_models(),
        required=True,
        help="Name of the model on huggingface."
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default="middle",
        help="Name of the model layer."
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs='+',
        choices=['all', 'idiom', 'literal', 'negative', 'neutral', 'positive'],
        required=True,
        help="List of conditions."
    )
    
    args = parser.parse_args()
    return args
        
if __name__ == '__main__':
    #TODO: make the arg parsing more user-friendly with entering name arguments. Entering the exact model names and layer names is cumbersome
    args = parse_arguments()
    
    model_name = args.model_name
    if args.layer_name == 'middle':
        layer_name = get_middle_layer(model_name)
    else:
        layer_name = args.layer_name
    conditions = args.conditions
    
    model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
    
    for condition in tqdm(conditions, desc='Conditions'):
        run_and_save(model_name, layer_name, condition, model=model)

"""
sbatch scripts/slurm_runner.sh distilgpt2 all idiom literal

sbatch scripts/slurm_runner.sh meta-llama/Llama-3.2-1B all idiom literal
sbatch scripts/slurm_runner.sh meta-llama/Llama-3.2-1B negative neutral positive

sbatch scripts/slurm_runner.sh distilgpt2 neutral positive

sbatch scripts/slurm_runner.sh DiscoResearch/Llama3-German-8B negative neutral
sbatch scripts/slurm_runner.sh DiscoResearch/Llama3-German-8B positive

sbatch scripts/slurm_runner.sh TUM/GottBERT_base_last idiom literal
sbatch scripts/slurm_runner.sh TUM/GottBERT_base_last neutral positive
"""