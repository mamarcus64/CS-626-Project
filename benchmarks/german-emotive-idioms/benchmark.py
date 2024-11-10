import xarray as xr

from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language import benchmark_registry


class GermanEmotiveIdioms(BenchmarkBase):

    def __init__(self):
        self.data = load_dataset('german-emotive-idioms')    # Replace
        self.metric = load_metric('pearsonr')

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI
        )
        stimuli = self.data['stimulus']
        predictions = candidate.digest_text(stimuli.values)['neural']
        raw_score = self.metric(predictions, self.data)
        score = ceiling_normalize(raw_score, self.ceiling)
        return score

benchmark_registry['german-emotive-idioms'] = GermanEmotiveIdioms