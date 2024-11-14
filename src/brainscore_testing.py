# change as needed
PATH_TO_BRAINSCORE_REPO = '/scratch1/mjma/CS-626-Project/brain-score-language/brainscore_language'

import sys
sys.path.append(PATH_TO_BRAINSCORE_REPO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pdb

from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})

pdb.set_trace()

from tqdm import tqdm
from brainio.assemblies import merge_data_arrays
from brainscore_language import load_benchmark

benchmark = load_benchmark('Pereira2018.243sentences-linear')

layer_scores = []
for layer in tqdm([f'transformer.h.{block}.ln_1' for block in range(6)], desc='layers'):
    layer_model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: layer})
    layer_score = benchmark(layer_model)
    # package for xarray
    layer_score = layer_score.expand_dims('layer')
    layer_score['layer'] = [layer]
    layer_scores.append(layer_score)
layer_scores = merge_data_arrays(layer_scores)
print(layer_scores)

pdb.set_trace()

import numpy as np
from matplotlib import pyplot

fig, ax = pyplot.subplots()
x = np.arange(len(layer_scores))
ax.scatter(x, layer_scores)
ax.set_xticks(x)
ax.set_xticklabels(layer_scores['layer'].values, rotation=90)
ax.set_ylabel('score')

fig.savefig('tmp/scores.png')