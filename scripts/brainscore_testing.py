# Include the following lines at the top of each file for file pathing.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # adjust relative path as needed
import file_utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# imports
import pdb
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from brainio.assemblies import merge_data_arrays
import os
import json
import h5py
from transformers import AutoModelForCausalLM

from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject
from brainscore_language import load_benchmark
from brainscore_language.benchmarks.german_emotive_idioms import GermanEmotiveIdioms
from brainscore_language.benchmarks.german_emotive_idioms.benchmark import cka, svcca, rdm
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_core.metrics import Score
from brainscore_language.metrics.neural_cosine_similarity.metric import NeuralCosineSimilarity


# h5_name, ceiling_name = 'fMRI_masked_semantic_data.h5', 'fMRI_masked_semantic_ceilings.json'
# h5_name, ceiling_name = 'fMRI_masked_sentences_data.h5', 'fMRI_masked_semantic_ceilings.json'
h5_name, ceiling_name = 'fMRI_semantic_all_language_areas_data.h5', 'fMRI_masked_semantic_ceilings.json'
# h5_name, ceiling_name = 'fMRI_semantic_MFG_data.h5', 'fMRI_masked_semantic_ceilings.json'
# h5_name, ceiling_name = 'fMRI_semantic_MTG_IFG_SMA_data.h5', 'fMRI_masked_semantic_ceilings.json'

# model_name = 'distilgpt2'
model_name = 'TUM/GottBERT_base_last'
# model_name = 'DiscoResearch/Llama3-German-8B'
# model_name = 'meta-llama/Llama-3.2-1B'

if len(sys.argv) >= 2:
    model_name = sys.argv[1]

# idiom_type = 'idiom'
# idiom_type = 'literal'
idiom_type = None


neural_data = None
with h5py.File(os.path.join(file_utils.VOXEL_PATH, h5_name), 'r') as f:
    neural_data = f['fMRI_sentences'][:]
    
standardize = True
if standardize:  # mean=0, std=1
    mean = np.mean(neural_data)
    std = np.std(neural_data)
    neural_data = (neural_data - mean) / std
    
num_subjects = neural_data.shape[1]

transposed_neural_data = np.transpose(neural_data, axes=(2, 0, 1))
reshaped_neural_data = transposed_neural_data.reshape(neural_data.shape[2], -1)

ceilings_file = os.path.join(file_utils.CEILING_PATH, ceiling_name)
with open(ceilings_file, "r") as f:
    ceiling = json.load(f)

# print(f"Neural data: {reshaped_neural_data.shape}")
# print(f"Ceilings: {list(ceiling.keys())}")


# Cast ceiling to Score
score = ceiling['score']
raw = ceiling['raw']
ceiling = Score(score)
ceiling.raw = raw
ceiling.name = 'data'

# pick metric
metric = None # defaults to linear pearson correlation
# metric = cka
# metric = NeuralCosineSimilarity()

if idiom_type is None:
    selected_neural_data = reshaped_neural_data
    selected_stimuli_ids = list(range(1, 181)) # "Code" field
elif idiom_type == 'idiom':
    selected_neural_data = reshaped_neural_data[:90]
    selected_stimuli_ids = list(range(1, 91))
elif idiom_type == 'literal':
    selected_neural_data = reshaped_neural_data[90:]
    selected_stimuli_ids = list(range(91, 181))
    
benchmark = GermanEmotiveIdioms(
    neural_data=selected_neural_data,
    num_subjects=num_subjects,
    selected_stimuli_ids=selected_stimuli_ids,
    ceiling=ceiling,
    metric=metric,
    )
data = benchmark.data
df = benchmark.data.to_dataframe()

# pdb.set_trace()
# might need to log into huggingface beforehand (try huggingface-cli login)


if model_name == 'distilgpt2':
    # layer_names = [f'transformer.h.{block}.ln_1' for block in range(6)]
    layer_names = [f'transformer.h.{block}.ln_1' for block in range(3, 6)]
elif model_name == 'TUM/GottBERT_base_last':
    layer_names = [f'roberta.encoder.layer.{block}.attention' for block in range(6, 9)]
    # layer_names = ['roberta.embeddings.LayerNorm'] + [f'roberta.encoder.layer.{block}.output.LayerNorm' for block in range(12)]
    
#TODO: fix the errors for LLaMA models. Weight mismatch when loading from huggingface
elif model_name == 'DiscoResearch/Llama3-German-8B':
    layer_names = [f'model.layers.{block}.mlp.gate_proj' for block in range(16, 20)]
elif model_name == 'meta-llama/Llama-3.2-1B':
    # layer_names = ['model.embed_tokens'] + [f'model.layers.{block}.mlp.gate_proj' for block in range(16)]
    layer_names = [f'model.layers.{block}.mlp.gate_proj' for block in range(8, 11)]
    

model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
print(model)

print(h5_name, ceiling_name, model_name)

layer_scores = []
for i, layer in tqdm(enumerate(layer_names), desc='layers', total=len(layer_names)):
    layer_model = HuggingfaceSubject(model_id=model_name, model=model, region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: layer})
    layer_score = benchmark(layer_model, i, save_folder=os.path.join(file_utils.PARENT_PATH, f'{model_name.replace("/", "_")}'))
    print(layer_score)
    # package for xarray
    layer_score = layer_score.expand_dims('layer')
    layer_score['layer'] = [layer]
    layer_scores.append(layer_score)

print('done generating layer scores')
    
layer_scores = merge_data_arrays(layer_scores)
print(layer_scores)

fig, ax = plt.subplots()
x = np.arange(len(layer_scores))
ax.scatter(x, layer_scores)
ax.set_xticks(x)
ax.set_xticklabels(layer_scores['layer'].values, rotation=90)
fig.savefig(f'tmp/{model_name.replace("/", "_")}_scores.png')


pdb.set_trace()
