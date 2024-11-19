from brainscore_language import benchmark_registry
from .benchmark import GermanEmotiveIdioms

# TODO: Change metric name after Nona's implementation
benchmark_registry['GermanEmotiveIdioms-pearsonr'] = GermanEmotiveIdioms
