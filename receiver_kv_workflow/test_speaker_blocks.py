import unittest
from unittest.mock import patch
from .speaker_blocks import speaker_ranges, speaker_units
from .detail_selection import select_complete_units


class SpeakerBlockTests(unittest.TestCase):
    def test_consecutive_same_speaker_but_not_later_return(self):
        text = "Meeting\nAlice (Team)  0:01\nFirst.\nAlice (Team)  0:05\nCorrection.\nBob  0:10\nAgreed.\nAlice (Team)  0:12\nNext."
        blocks = speaker_ranges(text)
        self.assertEqual([b['speaker'] for b in blocks], [None, 'Alice (Team)', 'Bob', 'Alice (Team)'])
        self.assertIn('Correction.', text[blocks[1]['start']:blocks[1]['end']])
        self.assertEqual(''.join(text[b['start']:b['end']] for b in blocks), text)

    def test_colon_and_timestamp_labels(self):
        text = "[10:30] Alice: First\nALICE: Second\nBob: Third"
        self.assertEqual(len(speaker_ranges(text)), 2)
        with self.assertRaisesRegex(ValueError, 'speaker labels'):
            speaker_ranges('Unlabelled meeting transcript.')

    def test_whole_blocks_cover_source_and_obey_budget(self):
        text = "Alice: First.\nAlice: Second.\nBob: Agreed."
        offsets = [(i, i + 1) for i in range(len(text))]
        with patch('receiver_kv_workflow.speaker_blocks.source_offsets', return_value=(offsets, 0)):
            units = speaker_units(None, text, [], [])
        self.assertEqual([i for unit in units for i in unit], list(range(len(text))))
        selected = select_complete_units(units, list(range(15)), [[1.0] * len(text)], [])
        self.assertLessEqual(len(selected), 15)
        for unit in units:
            self.assertIn(len(set(unit) & set(selected)), (0, len(unit)))


if __name__ == '__main__':
    unittest.main()
