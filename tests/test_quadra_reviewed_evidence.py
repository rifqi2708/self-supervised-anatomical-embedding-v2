import json
import tempfile
import unittest
from pathlib import Path

from tools.quadra import aligned_organ_group_cohort as cohort


def fixture_contract(root):
    root.mkdir()
    rows = [dict(query_id='q'+str(i), subject_id='s1', mask_name='liver',
                 group_name='abdomen', physical_lps_x=i, physical_lps_y=0,
                 physical_lps_z=0) for i in range(1, 5)]
    cohort.atomic_csv(root/'frozen_queries_raw_itk.csv', rows)
    path = root/'frozen_queries_raw_itk.csv'
    cohort.atomic_json(root/'matching_contract.json', dict(schema_version=2,
        fixture_only=True, dataset_id='synthetic-reviewed', query_count=4,
        subjects=[dict(subject_id='s1', review_partition='discovery')],
        files=[dict(path=path.name, bytes=path.stat().st_size,
                    sha256=cohort.sha256_file(path))]))
    return root


class ReviewedEvidenceTests(unittest.TestCase):
    def test_resume_preserves_terminal_results_and_seals_staged_artifacts(self):
        from tools.quadra import reviewed_evidence as evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); contract = fixture_contract(root/'contract')
            first = dict(query_id='q1',status='success',forward_status='success',
                reverse_status='success',cycle_error_mm=1)
            run = root/'run'
            evidence.write_method_bundle(contract,run,'registration',[first])
            stage = root/'stage'; (stage/'registration').mkdir(parents=True)
            (stage/'registration'/'transform.json').write_text('{"fixture":true}')
            second = dict(first,query_id='q2',cycle_error_mm=2)
            evidence.write_method_bundle(contract,run,'registration',[first,second],
                resume=True,artifact_source=stage)
            manifest, rows = evidence.read_method_bundle(run,contract)
            self.assertEqual(manifest['attempted_queries'],2)
            self.assertEqual(manifest['status'],'partial')
            self.assertTrue((run/'registration'/'transform.json').is_file())
            with self.assertRaises(cohort.CohortError):
                evidence.write_method_bundle(contract,run,'registration',
                    [dict(first,cycle_error_mm=99),second],resume=True)
            manifest, rows = evidence.read_method_bundle(run,contract)
            self.assertEqual(rows[0]['cycle_error_mm'],'1')

    def test_rejected_staging_cannot_invalidate_previous_checkpoint(self):
        from tools.quadra import reviewed_evidence as evidence
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); contract=fixture_contract(root/'contract')
            stage=root/'stage'; stage.mkdir()
            (stage/'z-existing.json').write_text('original')
            first=dict(query_id='q1',status='success',forward_status='success',
                reverse_status='success',cycle_error_mm=1)
            run=root/'run'
            evidence.write_method_bundle(contract,run,'registration',[first],artifact_source=stage)
            (stage/'a-new.json').write_text('new')
            (stage/'z-existing.json').write_text('conflict')
            with self.assertRaises(cohort.CohortError):
                evidence.write_method_bundle(contract,run,'registration',[first],resume=True,artifact_source=stage)
            manifest,rows=evidence.read_method_bundle(run,contract)
            self.assertEqual(manifest['attempted_queries'],1)
            self.assertFalse((run/'a-new.json').exists())


    def test_comparison_uses_distinct_pairwise_shared_valid_sets(self):
        from tools.quadra import reviewed_evidence as evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); contract = fixture_contract(root/'contract')
            runs = []
            for method, errors in [('uae_nn',[1,2,100,None]),
                                   ('uae_fixed_point',[0,None,80,4]),
                                   ('registration',[None,1,90,2])]:
                records = []
                for i, error in enumerate(errors,1):
                    records.append(dict(query_id='q'+str(i),status='failed' if error is None else 'success',
                        forward_status='failed' if error is None else 'success',
                        reverse_status='not_attempted' if error is None else 'success',
                        failure_reason='fixture_failure' if error is None else '',
                        cycle_error_mm='' if error is None else error))
                run = root/method; runs.append(run)
                evidence.write_method_bundle(contract,run,method,records)
            rules = root/'rules.json'
            rules.write_text(json.dumps(dict(status='provisional',quantile='linear',
                organ_weighting='query_pool_within_subject_group',minimum_valid_median=1,
                minimum_valid_p95=2,inference='none')))
            argv = ['reviewed-compare','--contract',str(contract),
                    '--output-directory',str(root/'analysis'),'--rules',str(rules)]
            for run in runs: argv += ['--run-directory',str(run)]
            self.assertEqual(cohort.main(argv),0)
            rows = cohort.read_csv(root/'analysis'/'paired_differences.csv')
            values = {(r['method'],r['metric']):r for r in rows}
            self.assertAlmostEqual(float(values[('uae_fixed_point','median')]['difference_mm']),-10.5)
            self.assertAlmostEqual(float(values[('registration','median')]['difference_mm']),-5.5)
            self.assertAlmostEqual(float(values[('uae_fixed_point','p95')]['difference_mm']),-19.05)
            self.assertAlmostEqual(float(values[('registration','p95')]['difference_mm']),-9.55)
            self.assertEqual(values[('uae_fixed_point','median')]['shared_query_ids'],'q1;q3')
            self.assertEqual(values[('registration','median')]['shared_query_ids'],'q2;q3')
            self.assertTrue((root/'analysis'/'paired_differences.png').is_file())
            self.assertEqual(len(list((root/'analysis').glob('*.png'))),1)
            absolute=cohort.read_csv(root/'analysis'/'absolute_errors_and_failures.csv')
            self.assertTrue(all(r['attempted_queries']=='4' and r['failed_queries']=='1'
                and float(r['failure_rate_all_attempted'])==.25 for r in absolute))

    def test_intake_verifies_all_members_and_preserves_conflicts(self):
        from tools.quadra import reviewed_evidence as evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = fixture_contract(root/'contract')
            outcomes = [dict(query_id='q'+str(i), status='success',
                forward_status='success', reverse_status='success',
                cycle_error_mm=i) for i in range(1, 5)]
            run = root/'remote'
            evidence.write_method_bundle(contract, run, 'uae_nn', outcomes)
            self.assertEqual(cohort.main(['reviewed-intake', '--contract', str(contract),
                '--source', str(run), '--output-directory', str(root/'intake')]), 0)
            receipt = json.loads((root/'intake'/'intake_receipt.json').read_text())
            self.assertEqual(receipt['verification'], 'checksum_verified')
            self.assertEqual(receipt['method'], 'uae_nn')
            with (run/'query_outcomes.csv').open('a') as handle:
                handle.write('corruption\n')
            self.assertNotEqual(cohort.main(['reviewed-intake', '--contract', str(contract),
                '--source', str(run), '--output-directory', str(root/'bad')]), 0)
            self.assertTrue(run.exists())
            self.assertFalse((root/'bad').exists())
