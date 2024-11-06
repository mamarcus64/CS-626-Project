import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.append("../.venv")
sys.path.append("../brain-score-language/brainscore_language")

from brainio.assemblies import merge_data_arrays
from brainscore_language import load_benchmark
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

from tqdm import tqdm


if __name__ == "__main__":
    model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
    print(model.basemodel)

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