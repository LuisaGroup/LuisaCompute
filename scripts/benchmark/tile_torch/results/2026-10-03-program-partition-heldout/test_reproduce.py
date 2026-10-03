"""Host-only regression tests against the public captured receipt projection."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import reproduce as r

class PublicReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=Path(__file__).resolve().parent
        cls.original=json.loads((cls.folder/'validation.json').read_text(encoding='utf-8'))

    def test_all_samples_counts_and_negative_observations_preserved(self):
        audit=r.validate(self.original)
        self.assertEqual((audit['native_executions'],audit['torch_executions'],audit['samples']),(50,12,434))
        self.assertEqual(len(audit['selected_regressions']),2)
        self.assertEqual(audit,json.loads((self.folder/'audit.json').read_text(encoding='utf-8')))

    def test_policy_ties_and_known_geometry_without_refitting(self):
        self.assertEqual(r.prediction(31,1024,8)['selected_rows'],1)
        self.assertEqual(r.prediction(1,128,4)['status'],'retained')
        d=r.prediction(257,128,8)
        self.assertEqual((d['status'],d['selected_rows']),('selected',1))
        self.assertEqual(r.prediction(24,128,1)['selected_rows'],1)

    def test_raw_sample_mutation_invalidates_median(self):
        data=copy.deepcopy(self.original)
        data['cases'][0]['cohorts']['policy']['routes']['native']['event_us']['samples']=[1.]*7
        with self.assertRaisesRegex(ValueError,'median'):r.validate(data)

    def test_ratio_mutation_rejected(self):
        data=copy.deepcopy(self.original)
        data['cases'][0]['comparisons']['policy']['candidate_over_default']=1.
        with self.assertRaisesRegex(ValueError,'ratio'):r.validate(data)

    def test_source_prefix_mutation_rejected(self):
        data=copy.deepcopy(self.original)
        next(iter(data['cases'][0]['cohorts']['policy']['source_files'].values()))['original_source_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'source'):r.validate(data)

    def test_model_fit_or_decision_change_rejected(self):
        for key,value in (('fit','different-fit'),('selected_rows',8),('original_score',123.)):
            data=copy.deepcopy(self.original)
            data['cases'][0]['cohorts']['policy']['policy_decisions'][0][key]=value
            with self.assertRaises(ValueError):r.validate(data)

    def test_retry_cannot_fill_initial_torch_denominator(self):
        data=copy.deepcopy(self.original)
        row=next(x for x in data['cases'] if 'independent_torch_retest' in x)
        row['cohorts']['default']['routes']['torch']=row['independent_torch_retest']['evidence']['routes']['torch']
        with self.assertRaises((ValueError,KeyError)):r.validate(data)

    def test_retry_fixture_change_rejected(self):
        data=copy.deepcopy(self.original)
        row=next(x for x in data['cases'] if 'independent_torch_retest' in x)
        row['independent_torch_retest']['evidence']['fixture_sha256']={'different':'bytes'}
        with self.assertRaisesRegex(ValueError,'fixture'):r.validate(data)

    def test_original_failure_and_schema_failure_cannot_be_erased(self):
        data=copy.deepcopy(self.original)
        data['native_revalidation']['original_cohort_status']='passed'
        with self.assertRaisesRegex(ValueError,'failure'):r.validate(data)
        data=copy.deepcopy(self.original)
        del data['preserved_schema_failure']
        with self.assertRaisesRegex(ValueError,'provenance'):r.validate(data)

    def test_public_byte_receipts_detect_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);file=folder/'example.txt';file.write_bytes(b'original\n')
            (folder/'receipts.json').write_text(json.dumps({'files':{'example.txt':{'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':9}}}))
            r.check_receipts(folder)
            file.write_bytes(b'changed!\n')
            with self.assertRaisesRegex(ValueError,'bytes'):r.check_receipts(folder)

if __name__=='__main__':unittest.main()
