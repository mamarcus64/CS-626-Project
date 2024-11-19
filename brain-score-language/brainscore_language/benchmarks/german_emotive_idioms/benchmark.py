import numpy as np
import os
import pandas as pd
import sys
import xarray as xr

from brainio.assemblies import DataAssembly, NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score, Metric

# Append local brain-score repo to path, brittle but should be fine
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "brain-score-language")))
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language import benchmark_registry

from scipy.stats import pearsonr


def _build_id(assembly, coords):
    ids =  [".".join([f"{value}" for value in values]) for values in zip(*[assembly[coord].values for coord in coords])]
    return ids


def _build_id_from_subjects_and_voxels(voxel_subjects, voxel_nums):
    ids = [f"{voxel_subjects[i]}.{voxel_nums[i]}" for i in range(len(voxel_subjects))]
    return ids


class GermanEmotiveIdioms(BenchmarkBase):

    def __init__(self, neural_data, ceiling):
        num_subjects = 10
        voxels_per_subject = 15
        num_stimuli = 12

        stimuli_file = os.path.abspath(os.path.join(os.path.pardir, "emotive_idioms_dataset", "stimuli.tsv"))
        selected_stimuli_ids = list(range(1, num_stimuli + 1)) # "Code" field
        stimuli = pd.read_csv(stimuli_file, sep='\t')
        selected_stimuli = stimuli[stimuli['Code'].isin(selected_stimuli_ids)]
        sentences = selected_stimuli["Stimulus"].to_numpy().tolist()

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
        for i in range(1, num_subjects + 1): # TODO: need to make this more robust/matching other code, will do after sync with Helen
            voxel_subjects.extend([str(i)] * voxels_per_subject)
            voxel_nums.extend(list(range(voxels_per_subject)))
        
        neuroid_ids = _build_id_from_subjects_and_voxels(voxel_subjects, voxel_nums)
        self.data = NeuroidAssembly(
            neural_data, 
            coords={
                'stimulus_num': ('presentation', selected_stimuli_ids),
                'stimulus_id': ('presentation', [str(i) for i in selected_stimuli_ids]),
                'sentence': ('presentation', sentences),
                'stimulus': ('presentation', sentences),
                'subject': ('neuroid', voxel_subjects),
                'voxel_num': ('neuroid', voxel_nums),
                'neuroid_id': ('neuroid', neuroid_ids)
            },
            dims=['presentation', 'neuroid']
        )

        self.data.name = 'data'
        self.data.attrs['identifier'] = 'german_emotive_idioms'

        self.metric = load_metric('linear_pearsonr')

        super(GermanEmotiveIdioms, self).__init__(
            identifier="GermanEmotiveIdioms",
            version=1,
            parent='GermanEmotiveIdioms',
            ceiling=ceiling
        )

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI
        )
        stimuli = self.data['stimulus']
        predictions = candidate.digest_text(stimuli.values)['neural']
        predictions['stimulus_id'] = 'presentation', stimuli['stimulus_id'].values
        raw_score = self.metric(predictions, self.data)
        score = ceiling_normalize(raw_score, self.ceiling)
        return score
