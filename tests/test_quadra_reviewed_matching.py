import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from tools.quadra import aligned_organ_group_cohort as cohort
from tools.quadra import disposable_pod


class ReviewedMatchingCommandTests(unittest.TestCase):
    def reviewed_dataset(self, root):
        masks, cts = root / 'masks', root / 'cts'
        masks.mkdir(); cts.mkdir()
        registry = cohort.registry_records(cohort.PROJECT_ROOT / 'tools/quadra/totalsegmentator/organs.yaml')
        data = np.zeros((4, 4, 4), dtype=np.uint8)
        data[1, 1, 1] = data[2, 1, 1] = 1
        image = nib.Nifti1Image(data, np.eye(4))
        image.header.set_xyzt_units('mm')
        seed = root / 'seed.nii.gz'; nib.save(image, str(seed))
        payload = seed.read_bytes(); sha = hashlib.sha256(payload).hexdigest()
        rows = []
        for number in range(1, 49):
            subject = 'quadra_hc_{:03d}'.format(number)
            sex = 'F' if number <= 25 else 'M'
            for session in ('test', 'retest'):
                ct = cts / 'QUADRA_HC_{:03d}'.format(number) / (session + '_CT-AC.nii.gz')
                ct.parent.mkdir(exist_ok=True); ct.write_bytes(payload)
                target = masks / subject / session / 'masks'; target.mkdir(parents=True)
                for organ in registry:
                    name = organ['filename']
                    if name == 'prostate' and sex == 'F': continue
                    path = target / (name + '.nii.gz'); path.write_bytes(payload)
                    rows.append(dict(subject_id=subject, session=session, sex=sex, organ=name,
                                     dataset_mask_sha256=sha, ct_sha256=sha,
                                     selected_source='corrected_derivative' if number == 1 else 'original',
                                     acceptance_stage='post_correction_review' if number == 1 else 'first_segmentation'))
        with (masks / 'final_mask_inventory.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        manifest = dict(schema_version=1, dataset_kind='quadra_final_reviewed_masks_for_matching',
                        counts=dict(subjects=48, scans=96, masks=3790),
                        validation=dict(all_decisions_acceptable=True, copy_checksums_verified=True))
        (masks / 'final_dataset_manifest.json').write_text(json.dumps(manifest))
        return masks, cts

    def test_freeze_new_inventory_preserves_shortfalls_and_fine_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks, cts = self.reviewed_dataset(root)
            output = root / 'frozen'
            self.assertEqual(cohort.main(['freeze-reviewed', '--dataset-root', str(masks),
                                         '--ct-root', str(cts), '--output-directory', str(output)]), 0)
            manifest = json.loads((output / 'matching_contract.json').read_text())
            self.assertEqual(manifest['query_count'], 3790)
            self.assertEqual(manifest['counts']['scans'], 96)
            self.assertEqual(manifest['counts']['masks'], 3790)
            self.assertEqual(len(manifest['plan_files']), 384)
            self.assertEqual(len(manifest['registration_plan_files']), 384)
            self.assertEqual(sum(s['review_partition'] == 'confirmation' for s in manifest['subjects']), 16)
            discovery = {s['subject_id'] for s in manifest['subjects'] if s['review_partition'] == 'discovery'}
            self.assertTrue(set(manifest['pilot_subjects']) <= discovery)
            with (output / 'frozen_queries_raw_itk.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len({r['query_id'] for r in rows}), 3790)
            self.assertTrue(all(int(r['mask_query_shortfall']) == 98 for r in rows))
            first = [r for r in rows if r['subject_id'] == 'quadra_hc_001' and r['mask_name'] == 'brain']
            self.assertEqual(len(first), 2)
            self.assertEqual(len({(r['fine_x'], r['fine_y'], r['fine_z']) for r in first}), 1)
    def contract(self, root):
        root.mkdir()
        queries = root / 'frozen_queries_raw_itk.csv'
        queries.write_text('query_id,subject_id,mask_name,raw_x,raw_y,raw_z\nq1,s1,brain,1,2,3\nq2,s1,brain,2,2,3\n')
        payload = {
            'schema_version': 2, 'dataset_id': 'synthetic-reviewed',
            'fixture_only': True, 'subjects': ['s1'], 'query_count': 2,
            'files': [{'path': queries.name, 'bytes': queries.stat().st_size,
                       'sha256': hashlib.sha256(queries.read_bytes()).hexdigest()}],
        }
        (root / 'matching_contract.json').write_text(json.dumps(payload))
        return root

    def test_separate_method_runs_share_frozen_queries_without_launching_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.contract(root / 'inputs')
            manifests = []
            for method in ('uae_nn', 'uae_fixed_point', 'registration'):
                run = root / method
                self.assertEqual(cohort.main(['prepare-method', '--contract', str(contract),
                                             '--method', method, '--run-directory', str(run)]), 0)
                manifests.append(json.loads((run / 'method_manifest.json').read_text()))
                with (run / 'query_outcomes.csv').open() as handle:
                    self.assertEqual([r['query_id'] for r in csv.DictReader(handle)], ['q1', 'q2'])
            self.assertEqual(len({m['input_signature'] for m in manifests}), 1)
            self.assertEqual([m['method'] for m in manifests], ['uae_nn', 'uae_fixed_point', 'registration'])
            self.assertTrue(all(m['status'] == 'awaiting_pilot' and
                                not m['scientific_work_launched'] for m in manifests))

    def test_setup_package_uses_new_inputs_and_stops_before_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks, cts = self.reviewed_dataset(root)
            contract = root / 'contract'
            self.assertEqual(cohort.main(['freeze-reviewed', '--dataset-root', str(masks),
                '--ct-root', str(cts), '--output-directory', str(contract)]), 0)
            package = root / 'setup'
            self.assertEqual(disposable_pod.main(['reviewed-package', '--contract', str(contract),
                '--dataset-root', str(masks), '--output-directory', str(package)]), 0)
            catalog = json.loads((package / 'disposable-reviewed-assets.json').read_text())
            self.assertEqual(catalog['subjects'], dict(first=1, last=48, count=48))
            self.assertEqual(catalog['assets']['stage5_masks']['expected']['final_masks'], 3790)
            self.assertEqual(catalog['assets']['whole_body_ct']['expected']['promoted_ct_files'], 96)
            self.assertEqual(catalog['assets']['experiment_contract']['expected']['frozen_queries'], 3790)
            for profile in ('uae', 'registration'):
                plan = json.loads((package / (profile+'-setup-plan.json')).read_text())
                self.assertIn('--setup-only', plan['bootstrap_command'])
                self.assertFalse(plan['scientific_work_launched'])
            source = masks / 'quadra_hc_001/test/masks/brain.nii.gz'
            self.assertTrue(source.exists())
            verify = ['reviewed-verify-package', '--package-directory', str(package)]
            self.assertEqual(disposable_pod.main(verify), 0)
            with (package/'reviewed-masks.tar.gz').open('ab') as handle:
                handle.write(b'corruption')
            self.assertEqual(disposable_pod.main(verify), 2)

    def test_freeze_refuses_changed_masks_without_overwriting_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks, cts = self.reviewed_dataset(root)
            source = masks/'quadra_hc_001/test/masks/brain.nii.gz'
            source.write_bytes(b'changed input')
            output = root/'contract'
            self.assertEqual(cohort.main(['freeze-reviewed', '--dataset-root', str(masks),
                '--ct-root', str(cts), '--output-directory', str(output)]), 3)
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), b'changed input')

    def test_contract_rejects_inconsistent_ct_identity_even_if_inventory_is_resigned(self):
        from tools.quadra import reviewed_matching_contract as reviewed
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks, cts = self.reviewed_dataset(root)
            output = root/'contract'
            self.assertEqual(cohort.main(['freeze-reviewed', '--dataset-root', str(masks),
                '--ct-root', str(cts), '--output-directory', str(output)]), 0)
            path = output/'portable_mask_inventory.csv'
            rows = cohort.read_csv(path)
            rows[1]['ct_sha256'] = '0'*64
            cohort.atomic_csv(path, rows)
            manifest = cohort.load_json(output/'matching_contract.json')
            for record in manifest['files']:
                if record['path'] == path.name:
                    record.update(bytes=path.stat().st_size, sha256=cohort.sha256_file(path))
            cohort.atomic_json(output/'matching_contract.json', manifest)
            with self.assertRaises(cohort.CohortError):
                reviewed.read_contract(output)

    def test_fixture_method_exports_failures_and_resumes_without_duplicate_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.contract(root / 'inputs')
            run = root / 'registration'
            prepare = ['prepare-method', '--contract', str(contract), '--method', 'registration', '--run-directory', str(run)]
            self.assertEqual(cohort.main(prepare), 0)
            fixtures = root / 'fixture.json'
            fixtures.write_text(json.dumps([
                dict(query_id='q1', status='success', forward_status='success', reverse_status='success', cycle_error_mm=1.0),
                dict(query_id='q2', status='failed', forward_status='out_of_domain', reverse_status='not_attempted', failure_reason='out_of_domain')]))
            replay = ['fixture-method', '--contract', str(contract), '--run-directory', str(run), '--outcomes', str(fixtures)]
            self.assertEqual(cohort.main(replay), 0)
            self.assertEqual(cohort.main(replay+['--resume']), 0)
            with (run/'query_outcomes.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r['status'] for r in rows], ['success', 'failed'])
            manifest = json.loads((run/'method_manifest.json').read_text())
            self.assertEqual((manifest['attempted_queries'], manifest['failed_queries']), (2, 1))
            self.assertFalse(manifest['scientific_work_launched'])
            (contract/'frozen_queries_raw_itk.csv').write_text('stale\n')
            self.assertEqual(cohort.main(prepare+['--resume']), 3)


if __name__ == '__main__':
    unittest.main()
