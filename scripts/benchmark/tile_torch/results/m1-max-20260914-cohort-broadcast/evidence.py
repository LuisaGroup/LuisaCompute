"""Read logical archived paths from deduplicated SHA256 shards, without extraction."""
import hashlib
import json
from pathlib import Path
import tarfile


class Evidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.cache = {}
        self.loaded = set()

    def alias(self, original):
        value = str(original)
        if value.startswith('/tmp/'):
            value = '/private' + value
        return self.manifest['aliases'][value]

    def blob(self, digest):
        info = self.manifest['objects'][digest]
        name = info['shard']
        if name not in self.loaded:
            with tarfile.open(self.directory / name) as archive:
                actual = set()
                for member in archive:
                    assert member.isfile() and member.name not in actual
                    actual.add(member.name)
                    expected = self.manifest['objects'][member.name]
                    value = archive.extractfile(member).read()
                    assert expected['shard'] == name and len(value) == expected['size']
                    assert hashlib.sha256(value).hexdigest() == member.name
                    self.cache[member.name] = value
            assert actual == {key for key, entry in self.manifest['objects'].items() if entry['shard'] == name}
            self.loaded.add(name)
        return self.cache[digest]

    def read(self, name):
        info = self.manifest['files'][name]
        result = self.blob(info['sha256'])
        assert len(result) == info['size']
        return result

    def json(self, name):
        return json.loads(self.read(name))

    def digest(self, name):
        self.read(name)
        return self.manifest['files'][name]['sha256']

    def close(self):
        self.cache.clear()
        self.loaded.clear()


def recompute(store):
    import math
    import statistics as st
    declared = store.json('cohort/plan_v1.json')
    report = store.json('cohort/replay-results.json')
    assert report['status'] == 'passed' and len(report['jobs']) == 44
    expected = {(case[0], local, arm) for case in declared['cases'] for local in (1, 8)
                for arm in ('disable-uniform', 'disable-cohort')}
    actual, output = set(), []
    for job in report['jobs']:
        name, local, arm = job['case'], job['local_lanes'], job['arm']
        key = name, local, arm
        assert key not in actual and job['status'] == 'passed'
        actual.add(key)
        replay = store.json(f'cohort/replayed-{name}-l{local}-{arm}/results.json')
        assert replay['status'] == 'passed' and replay['artifacts_unchanged'] and len(replay['visits']) == 12
        options = replay['options']
        assert [options[k] for k in ('cycles', 'samples', 'warmup_ms', 'target_ms')] == [3, 5, 40, 20]
        values, groups = [], {'default': [], arm: []}
        for i, row in enumerate(replay['visits']):
            assert row['cycle'] == i // 4 and row['position'] == i % 4
            assert row['variant'] == ('default', arm, arm, 'default')[i % 4]
            assert row['valid'] and row['returncode'] == 0 and row['all_guards_passed'] and row['inputs_unchanged']
            samples = row['samples_us']
            assert len(samples) == 5 and all(math.isfinite(v) and v > 0 for v in samples)
            value = st.median(samples)
            assert value == row['median_us'] and row['repetitions'] > 0
            values.append(value)
            groups[row['variant']].append(value)
        medians = {variant: st.median(values) for variant, values in groups.items()}
        pairs = [values[4 * c + b] / values[4 * c + a] for c in range(3) for a, b in ((0, 1), (3, 2))]
        statistics = dict(baseline='default', candidate=arm, pairs=pairs, median=st.median(pairs), minimum=min(pairs), maximum=max(pairs))
        assert medians == replay['summary_us'] == job['summary_us']
        assert statistics == replay['candidate_over_baseline'] == job['disabled_over_default']
        output.append(dict(case=name, local_lanes=local, disabled_option=arm, median_us=medians,
                           disabled_over_default=statistics, default_faster_pairs=sum(v > 1 for v in pairs)))
    assert actual == expected
    return dict(status='passed', producer='full9 / cohort-recurrences', cases=declared['cases'], jobs=output,
                captures=66, prepares=66, paired_replays=44, visits=528, samples=2640,
                metric='single_thread_native_entry_host_wall_us', ratio='disabled/default; greater than 1 means default faster',
                boundary='Two independent A/B edges, not a complete 2x2; never multiply edges or compare with other timers/Torch/MPS.')


if __name__ == '__main__':
    import sys
    store = Evidence(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
    print(json.dumps(recompute(store), indent=2))
    store.close()
