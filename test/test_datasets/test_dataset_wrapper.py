# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset


class TestDistributedRepeatingDatasetStatistics(unittest.TestCase):

    def _make_wrapper(self, dim=None):
        wrapper = DistributedRepeatingDataset.__new__(
            DistributedRepeatingDataset)
        wrapper.statistic_name = 'private'
        wrapper.dim = dim
        return wrapper

    def test_combines_weighted_mean_and_std_with_scalar_counts(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 8.0],
                        'max': [2.0, 12.0],
                        'mean': [1.0, 10.0],
                        'std': [0.5, 1.0],
                        'count': 2,
                        'q01': [0.0, 8.0],
                        'q99': [2.0, 12.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [3.0, 11.0],
                        'max': [7.0, 17.0],
                        'mean': [5.0, 14.0],
                        'std': [1.5, 2.0],
                        'count': 6,
                        'q01': [3.0, 11.0],
                        'q99': [7.0, 17.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [4.0, 13.0])
        np.testing.assert_allclose(combined['std'], [np.sqrt(4.75), 2.5])
        np.testing.assert_allclose(combined['q01'], [2.25, 10.25])
        np.testing.assert_allclose(combined['q99'], [5.75, 15.75])

    def test_unweighted_std_includes_between_dataset_variance(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0],
                        'max': [0.0],
                        'mean': [0.0],
                        'std': [0.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0],
                        'max': [10.0],
                        'mean': [10.0],
                        'std': [0.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [5.0])
        np.testing.assert_allclose(combined['std'], [5.0])

    def test_combines_weighted_mean_and_std_with_vector_counts(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 10.0],
                        'max': [0.0, 10.0],
                        'mean': [0.0, 10.0],
                        'std': [0.0, 0.0],
                        'count': [1, 9],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0, 20.0],
                        'max': [10.0, 20.0],
                        'mean': [10.0, 20.0],
                        'std': [0.0, 0.0],
                        'count': [9, 1],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [9.0, 11.0])
        np.testing.assert_allclose(combined['std'], [3.0, 3.0])

    def test_incomplete_counts_fall_back_to_unweighted_merge(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0],
                        'max': [0.0],
                        'mean': [0.0],
                        'std': [0.0],
                        'count': 100,
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0],
                        'max': [10.0],
                        'mean': [10.0],
                        'std': [0.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [5.0])
        np.testing.assert_allclose(combined['std'], [5.0])

    def test_padding_applies_to_quantiles_and_vector_counts(self):
        wrapper = self._make_wrapper(dim=4)
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 1.0, 2.0],
                        'max': [0.0, 1.0, 2.0],
                        'mean': [0.0, 1.0, 2.0],
                        'std': [0.0, 0.0, 0.0],
                        'count': [1, 1, 1],
                        'q25': [0.0, 1.0, 2.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [4.0, 5.0, 6.0],
                        'max': [4.0, 5.0, 6.0],
                        'mean': [4.0, 5.0, 6.0],
                        'std': [0.0, 0.0, 0.0],
                        'count': [3, 3, 3],
                        'q25': [4.0, 5.0, 6.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [3.0, 4.0, 5.0, 3.0])
        np.testing.assert_allclose(combined['q25'], [3.0, 4.0, 5.0, 3.0])


if __name__ == '__main__':
    unittest.main()
