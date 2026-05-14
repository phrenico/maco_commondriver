import unittest
from pathlib import Path

from scripts.experiments.experiment_registry import (FAMILY_SPECS,
                                               get_execution_steps,
                                               get_family_spec,
                                               get_plot_family_specs)


class TestExperimentRegistry(unittest.TestCase):
    def test_expected_families_present(self):
        self.assertEqual(set(FAMILY_SPECS), {'logmaps', 'tentmaps', 'lorenz', 'lorenz_htune'})

    def test_combined_csv_names_match_current_outputs(self):
        self.assertEqual(get_family_spec('logmaps').combined_csv, 'logmaps_res.csv')
        self.assertEqual(get_family_spec('tentmaps').combined_csv, 'tentmaps_res.csv')
        self.assertEqual(get_family_spec('lorenz').combined_csv, 'lorenz_res.csv')
        self.assertEqual(get_family_spec('lorenz_htune').combined_csv, 'htune.csv')

    def test_result_files_are_unique_within_family(self):
        for family_spec in FAMILY_SPECS.values():
            self.assertEqual(len(family_spec.result_files), len(set(family_spec.result_files)))

    def test_plot_family_order_uses_family_specs(self):
        self.assertEqual(tuple(spec.key for spec in get_plot_family_specs()),
                         ('logmaps', 'tentmaps', 'lorenz'))

    def test_kpca_remains_limited_to_logmaps_and_tentmaps(self):
        logmap_files = get_family_spec('logmaps').result_files
        tentmap_files = get_family_spec('tentmaps').result_files
        lorenz_files = get_family_spec('lorenz').result_files

        self.assertIn('kpca_res.csv', logmap_files)
        self.assertIn('kpca_res.csv', tentmap_files)
        self.assertNotIn('kpca_res.csv', lorenz_files)

    def test_execution_steps_include_combine_stage(self):
        logmap_steps = get_execution_steps('logmaps')
        self.assertEqual(logmap_steps[-1].label, 'Combine')
        self.assertEqual(logmap_steps[-1].module, 'scripts.experiments.logmaps.Z_combine_final_res')

    def test_execution_stage_selection(self):
        self.assertEqual(len(get_execution_steps('logmaps', stage='methods')), 11)
        self.assertEqual(len(get_execution_steps('tentmaps', stage='methods')), 11)
        self.assertEqual(len(get_execution_steps('lorenz', stage='methods')), 9)
        self.assertEqual(len(get_execution_steps('lorenz_htune', stage='methods')), 4)
        self.assertEqual(len(get_execution_steps('lorenz', stage='combine')), 1)

    def test_all_registry_envs_have_env_specs(self):
        env_dir = Path(__file__).resolve().parents[1] / 'envs'
        self.assertTrue(env_dir.is_dir(), msg='Expected envs/ directory to exist')

        expected_envs = {
            step.env_name
            for family_key in FAMILY_SPECS
            for step in get_execution_steps(family_key, stage='all')
        }
        available_env_specs = {path.stem for path in env_dir.glob('*.yml')}

        missing = expected_envs - available_env_specs
        self.assertFalse(missing, msg=f'Missing env spec files for: {sorted(missing)}')

    def test_env_specs_have_no_orphans_against_registry(self):
        env_dir = Path(__file__).resolve().parents[1] / 'envs'
        expected_envs = {
            step.env_name
            for family_key in FAMILY_SPECS
            for step in get_execution_steps(family_key, stage='all')
        }
        available_env_specs = {path.stem for path in env_dir.glob('*.yml')}

        orphan_specs = available_env_specs - expected_envs
        self.assertFalse(orphan_specs, msg=f'Orphan env specs not referenced by registry: {sorted(orphan_specs)}')