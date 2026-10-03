"""Validate the compact failed eligibility record; it has no speed ratios."""
from pathlib import Path
import hashlib
import json
import reproduce_automatic as replay

HERE = Path(__file__).resolve().parent


def validate(x):
    replay.require(x['schema'] == 'automatic-scan-eligibility-failure-public-v1' and x['status'] == 'queue_failed' and
                   x['metrics'] is None and x['ratios'] is None, 'failed eligibility gained performance conclusions')
    queue, aggregate = x['queue'], x['aggregate']
    replay.require(queue['status'] == aggregate['status'] == 'failed' and queue['finished'] and aggregate['finished'],
                   'failure closure changed')
    replay.require(len(queue['stages']) == 1 and queue['stages'][0]['name'] == 'default' and
                   queue['stages'][0]['returncode'] == 1 and queue['stages'][0]['automatic_cost'] is False,
                   'unrun policy/recheck claimed')
    replay.require(x['not_started'] == ['automatic', 'recheck'] and len(x['cases']) == 6 and
                   {r['case']['id'] for r in x['cases']} == {r['id'] for r in queue['plan']['cases']},
                   'eligibility inventory/stages changed')
    bad = [r for r in x['cases'] if r['status'] != 'passed']
    replay.require(len(bad) == 1 and bad[0]['case']['id'] == 'scan-eligibility-3x8192-bf16-br4', 'failure identity changed')
    for row in x['cases']:
        replay.require(row['automatic_requested'] is False and row['fixed_threads_requested'] == 0 and
                       row['torch_status'] == 'passed', 'wrong default requests/status')
        if row is not bad[0]:
            replay.require(row['native_status'] == 'passed' and row['native_returncode'] == 0, 'another native failure')
    failure = x['failed_native_result']
    replay.require(failure['status'] == 'compiler_failure' and 'tileiras failed' in failure['reason'] and
                   '0x00000005' in failure['reason'] and failure['dimensions'] == [3, 8192] and
                   failure['tile'] == [4, 8192, 1] and failure['precision'] == 'bf16' and
                   bad[0]['native_returncode'] == 1 and bad[0]['failure_kind'] == 'compiler_failure',
                   'compiler failure classification changed')
    return dict(status='preserved_queue_failure', default_cases=6, native_passed=5, native_compiler_failures=1,
                automatic_executed=False, ratios_produced=False)


def main():
    manifest = json.loads((HERE / 'manifest-eligibility-failure.json').read_text(encoding='utf-8'))
    for name, r in manifest['files'].items():
        replay.require(Path(name).name == name, 'manifest escapes package')
        b = (HERE/name).read_bytes()
        replay.require(len(b) == r['bytes'] and hashlib.sha256(b).hexdigest() == r['sha256'], 'public bytes changed: '+name)
    print(json.dumps(validate(json.loads((HERE/'eligibility-failure.json').read_text(encoding='utf-8')))))


if __name__ == '__main__':
    main()
