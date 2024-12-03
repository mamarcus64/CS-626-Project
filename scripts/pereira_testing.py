# Include the following lines at the top of each file for file pathing.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # adjust relative path as needed
import file_utils

import numpy as np

from brainio.assemblies import merge_data_arrays
from brainscore_language import ArtificialSubject, load_benchmark
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

from matplotlib import pyplot as plt
from tqdm import tqdm

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