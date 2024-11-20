# change as needed
PATH_TO_BRAINSCORE_REPO = '/scratch1/mjma/CS-626-Project/brain-score-language/brainscore_language'

import sys
sys.path.append(PATH_TO_BRAINSCORE_REPO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# imports
import pdb
import numpy as np
from matplotlib import pyplot
from tqdm import tqdm
from brainio.assemblies import merge_data_arrays
import os
import json

from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject
from brainscore_language import load_benchmark
from brainscore_language.benchmarks.german_emotive_idioms import GermanEmotiveIdioms
from brainscore_language.benchmarks.german_emotive_idioms.benchmark import cka, svcca, rdm
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_core.metrics import Score

# might need to log into huggingface beforehand (try huggingface-cli login)
gpt_model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
bert_model = HuggingfaceSubject(model_id='TUM/GottBERT_base_last', region_layer_mapping={})

neural_data_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))),
                                  "..",
                                  "emotive_idioms_dataset",
                                  "processed_data")
neural_data_file = os.path.join(neural_data_folder, "dummy_data.npy")
ceilings_file = os.path.join(neural_data_folder, "dummy_ceilings.json")
neural_data = np.load(neural_data_file)
with open(ceilings_file, "r") as f:
    ceiling = json.load(f)

print(f"Neural data: {neural_data.shape}")
print(f"Ceilings: {list(ceiling.keys())}")
# print(ceiling)

# Cast ceiling to Score
score = ceiling['score']
raw = ceiling['raw']
ceiling = Score(score)
ceiling.raw = raw
ceiling.name = 'data'
print(ceiling)

metric = None
# metric = cka
benchmark = GermanEmotiveIdioms(neural_data, ceiling, metric=metric)
data = benchmark.data
df = benchmark.data.to_dataframe()
# print(data)
# print(data['presentation']['stimulus_id'])

# pdb.set_trace()

layer_scores = []
# for layer in tqdm([f'transformer.h.{block}.ln_1' for block in range(6)], desc='layers'):
for layer in tqdm([f'roberta.encoder.layer.{block}.output.dense' for block in range(12)], desc='layers'):
    layer_model = HuggingfaceSubject(model_id='TUM/GottBERT_base_last', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: layer})
    layer_score = benchmark(layer_model)
    # package for xarray
    layer_score = layer_score.expand_dims('layer')
    layer_score['layer'] = [layer]
    layer_scores.append(layer_score)
layer_scores = merge_data_arrays(layer_scores)
print(layer_scores)