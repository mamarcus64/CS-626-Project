import xarray as xr
import os
import sys
import pandas as pd

from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score

# append local brain-score repo to path, brittle but should be fine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "brain-score-language")))
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language import benchmark_registry


stimuli_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "stimuli.tsv"))
selected_stimuli_ids = list(range(1, 181)) # "Code" field

def _build_id(assembly, coords):
    return [".".join([f"{value}" for value in values]) for values in zip(*[assembly[coord].values for coord in coords])]

class GermanEmotiveIdioms(BenchmarkBase):

    def __init__(self, neural_data, selected_stimuli_ids=selected_stimuli_ids, stimuli_file=stimuli_file):
        stimuli = pd.read_csv(stimuli_file, sep='\t')
        selected_stimuli = stimuli[stimuli['Code'].isin(selected_stimuli_ids)]
        # seleted stimuli and provided neural data should be the same size
        assert neural_data.shape[0] == len(selected_stimuli_ids)
        """
        A Neuroid Assembly (https://github.com/brain-score/brainio/blob/main/brainio/assemblies.py)
        is just a wrapper for an XArray, which itself is basically a numpy array with metadata.
        For code compatability, the dimensions MUST be named exactly "presentation" and "neuroid",
        which correspond to the stimulus row and the neuron activation score column.
        """
        # assign each voxel to a subject
        voxel_subjects = []
        voxel_nums = []
        for i in range(1, 24): # TODO: need to make this more robust/matching other code, will do after sync with Helen
            voxel_subjects.extend([str(i)] * 2000)
            voxel_nums.extend(list(range(2000)))
        
        # TODO: add stimulus metadata as coords along the 'presentation' dimension
        self.data = NeuroidAssembly(neural_data, dims=['presentation', 'neuroid'])
        
        self.data['subject'] = ('neuroid', voxel_subjects)
        self.data['voxel_num'] = ('neuroid', voxel_nums)
        self.data['stimulus_id'] = ('presentation', selected_stimuli_ids)
        self.data['neuroid_id'] = ('neuroid', _build_id(self.data, ['subject', 'voxel_num']))
        self.data['presentation'] = ('presentation', selected_stimuli_ids)
        self.metric = load_metric('linear_pearsonr')

#     def __call__(self, candidate: ArtificialSubject) -> Score:
#         candidate.start_neural_recording(
#             recording_target=ArtificialSubject.RecordingTarget.language_system,
#             recording_type=ArtificialSubject.RecordingType.fMRI
#         )
#         stimuli = self.data['stimulus']
#         predictions = candidate.digest_text(stimuli.values)['neural']
#         raw_score = self.metric(predictions, self.data)
#         score = ceiling_normalize(raw_score, self.ceiling)
#         return score

# benchmark_registry['german-emotive-idioms'] = GermanEmotiveIdioms