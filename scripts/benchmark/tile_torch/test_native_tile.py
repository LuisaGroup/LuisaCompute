"""Native-entry LLM admission checks; no Torch, GPU or compiler required."""
import copy
from pathlib import Path
import tempfile
import unittest

import native_tile as replay


class LLMCaptureTest(unittest.TestCase):
    def metadata(self):
        return dict(operation='attention', dimensions=[1, 2, 1, 3, 7, 5, 4],
                    precision='fp32', fast_math=False, attention_block=[2, 3],
                    input_shapes=[[1, 2, 3, 5], [1, 1, 7, 5], [1, 1, 7, 4]],
                    output_shape=[1, 2, 3, 4], source_reduction_policy='unordered_tree',
                    relaxed_precision=False, requested_group_threads=0, requested_input_views=False, reduction_tree=False,
                    attention_qk='mma', attention_pv='mma')

    def test_decomposition_is_an_explicit_experiment_variable(self):
        left = self.metadata()
        right = copy.deepcopy(left)
        right.update(attention_qk='reduce', attention_pv='reduce')
        replay.check_matched_metadata(left, right, 'llm')

    def test_missing_or_unsupported_decomposition_fails(self):
        for field in ('attention_qk', 'attention_pv'):
            for mode in (None, 'automatic', True):
                right = self.metadata()
                right[field] = mode
                with self.assertRaisesRegex(ValueError, 'decomposition'):
                    replay.check_matched_metadata(self.metadata(), right, 'llm')

    def test_semantic_or_block_differences_fail(self):
        for key in ('operation', 'dimensions', 'precision', 'fast_math', 'attention_block',
                    'input_shapes', 'output_shape', 'source_reduction_policy', 'relaxed_precision',
                    'requested_group_threads', 'requested_input_views', 'reduction_tree'):
            right = self.metadata()
            right.pop(key)
            with self.assertRaisesRegex(ValueError, 'source case differs'):
                replay.check_matched_metadata(self.metadata(), right, 'llm')

    def test_three_inputs_are_preserved_with_exact_extents(self):
        import math
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'output.f32'
            metadata = self.metadata()
            for i, shape in enumerate(metadata['input_shapes']):
                Path(str(prefix) + f'.input{i}.f32').write_bytes(bytes(4 * math.prod(shape)))
            prefix.write_bytes(bytes(4 * math.prod(metadata['output_shape'])))
            inputs, output = replay.capture_payloads(prefix, metadata, 'llm')
            self.assertEqual(len(inputs), 3)
            self.assertEqual(output, prefix)
            inputs[2].write_bytes(b'\0' * 4)
            with self.assertRaisesRegex(ValueError, 'input byte extent'):
                replay.capture_payloads(prefix, metadata, 'llm')

    def test_invalid_shapes_rejected_before_read(self):
        for shape in ([], [True], [0], [-1], [2**27], ['4']):
            metadata = self.metadata()
            metadata['input_shapes'][0] = shape
            with self.assertRaisesRegex(ValueError, 'payload shape'):
                replay.capture_payloads(Path('/nonexistent'), metadata, 'llm')

    def test_legacy_two_input_convention_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'capture'
            Path(str(prefix) + '.input1.f32').write_bytes(bytes(4))
            inputs, output = replay.capture_payloads(prefix, {}, 'migrated')
            self.assertEqual(len(inputs), 2)
            self.assertEqual(output, Path(str(prefix) + '.output.f32'))

    def test_unknown_capture_kind_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsupported capture'):
            replay.capture_payloads(Path('/nonexistent'), {}, 'unknown')


if __name__ == '__main__':
    unittest.main()
