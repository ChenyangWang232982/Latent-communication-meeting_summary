import unittest

from experiment_1.components.length_groups import length_group


class LengthGroupsTest(unittest.TestCase):
    def test_half_open_boundaries_and_unknown_legacy_count(self):
        cases = {0: '0-4000', 3999: '0-4000', 4000: '4000-8000', 7999: '4000-8000',
                 8000: '8000-12000', 12000: '12000-16000', 15999: '12000-16000',
                 16000: '16000+', 100000: '16000+', None: 'unknown'}
        for count, group in cases.items():
            self.assertEqual(length_group(count), group)
