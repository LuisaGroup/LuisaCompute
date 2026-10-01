"""Host-only tests of the actual CUDA graph timer with a fake CUDA surface."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cuda_torch_baseline import time_graph


class GraphTimingTests(unittest.TestCase):
    def run_graph(self, spans, *, batch=100, samples=3, sample_ms=100):
        calls, events = [], []
        tick = 0
        spans = iter(spans)

        def clock():
            nonlocal tick
            tick += 100_000
            return tick

        class Stream:
            cuda_stream = 7

            def wait_stream(self, other):
                calls.append(('wait',))

        class Graph:
            def replay(self):
                calls.append(('replay',))

        class Event:
            def __init__(self, enable_timing):
                assert enable_timing
                self.index = len(events)
                events.append(self)
                calls.append(('new_event', self.index))

            def record(self):
                calls.append(('record', self.index))

            def synchronize(self):
                calls.append(('event_sync', self.index))

            def elapsed_time(self, other):
                assert self.index == 0 and other.index == 1
                calls.append(('elapsed',))
                return next(spans)

        stream = Stream()
        cuda = SimpleNamespace(CUDAGraph=Graph, Stream=Stream, Event=Event,
            current_stream=lambda: stream, stream=lambda _: nullcontext(),
            graph=lambda *args, **kwargs: nullcontext(),
            synchronize=lambda: calls.append(('sync',)))
        with patch('cuda_torch_baseline.time.perf_counter_ns', side_effect=clock):
            result = time_graph(SimpleNamespace(cuda=cuda), lambda: calls.append(('invoke',)),
                                batch=batch, samples=samples, sample_ms=sample_ms, warmup_ms=1.)
        return result, calls, events

    def test_prime_before_warmup_and_full_replay_count_normalization(self):
        result, calls, events = self.run_graph([999., 2., 100., 275., 125., 10.])
        self.assertEqual(len(events), 2)
        first = calls.index(('replay',))
        self.assertEqual(calls[first-1], ('record', 0))
        self.assertEqual(calls[first+1:first+4], [('record', 1), ('event_sync', 1), ('elapsed',)])
        replay_positions = [i for i, call in enumerate(calls) if call == ('replay',)]
        self.assertGreater(replay_positions[1], calls.index(('elapsed',)))
        self.assertGreater(result['warmup_replays'], 0)
        self.assertEqual(result['prime_event_ms'], 999.)
        self.assertEqual([a['replays'] for a in result['calibration']], [1, 50])
        self.assertEqual(result['replays_per_sample'], 50)
        self.assertEqual(result['operations_per_sample'], 5000)
        self.assertTrue(result['calibration_target_reached'])
        self.assertEqual(result['event_span_ms']['samples'], [275., 125., 10.])
        # The two outliers remain in the samples and median; no filtering.
        self.assertEqual(result['event_us_per_operation']['samples'], [55., 25., 2.])
        self.assertEqual(result['event_us_per_operation']['median'], 25.)
        self.assertEqual(len(replay_positions), 1 + result['warmup_replays'] + 1 + 50 + 3*50)
        for span, per_op in zip(result['host_span_ms']['samples'], result['host_wall_us_per_operation']['samples']):
            self.assertEqual(per_op, span*1000/5000)
        self.assertEqual(calls.count(('invoke',)), 3+100)
        self.assertEqual(calls.count(('record', 0)), 1+2+3)
        self.assertEqual(calls.count(('record', 1)), 1+2+3)

    def test_already_long_graph_keeps_one_replay(self):
        result, _, _ = self.run_graph([999., 300., 350.], batch=4, samples=1)
        self.assertEqual(result['replays_per_sample'], 1)
        self.assertEqual(result['operations_per_sample'], 4)
        self.assertEqual(result['event_us_per_operation']['median'], 87500.)
        self.assertEqual(len(result['calibration']), 1)

    def test_logical_operation_cap_is_reported_without_claiming_target(self):
        result, _, _ = self.run_graph([999., .0001, .0001, .0001], batch=1000, samples=1)
        self.assertEqual(result['replay_cap'], 10000)
        self.assertEqual(result['replays_per_sample'], 10000)
        self.assertEqual(result['operations_per_sample'], 10_000_000)
        self.assertFalse(result['calibration_target_reached'])

    def test_four_attempt_limit_retains_last_measured_count(self):
        result, _, _ = self.run_graph([999., 79., 79., 79., 79., 79.], samples=1)
        self.assertEqual([a['replays'] for a in result['calibration']], [1, 2, 3, 4])
        self.assertEqual(result['replays_per_sample'], 4)
        self.assertFalse(result['calibration_target_reached'])
        self.assertEqual(result['event_us_per_operation']['samples'], [197.5])

    def test_invalid_priming_calibration_and_formal_spans_fail_closed(self):
        for invalid in (0., -1., float('nan'), float('inf')):
            for stage in ('prime', 'calibration', 'sample'):
                with self.subTest(value=invalid, stage=stage):
                    with self.assertRaisesRegex(ValueError, 'invalid CUDA graph event span'):
                        spans = dict(prime=[invalid], calibration=[999., invalid], sample=[999., 100., invalid])[stage]
                        self.run_graph(spans, samples=1)


if __name__ == '__main__':
    unittest.main()
