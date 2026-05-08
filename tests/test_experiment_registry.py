import unittest

from scripts.resgen.experiment_registry import (FAMILY_SPECS,
                                               get_execution_steps,
                                               get_family_spec,
                                               get_plot_family_specs)


class TestExperimentRegistry(unittest.TestCase):
    def test_expected_families_present(self):
        self.assertEqual(set(FAMILY_SPECS), {'logmaps', 'tentmaps', 'lorenzs'})

    def test_combined_csv_names_match_current_outputs(self):
        self.assertEqual(get_family_spec('logmaps').combined_csv, 'logmaps_res.csv')
        self.assertEqual(get_family_spec('tentmaps').combined_csv, 'tentmaps_res.csv')
        self.assertEqual(get_family_spec('lorenzs').combined_csv, 'lorenzs_res.csv')

    def test_result_files_are_unique_within_family(self):
        for family_spec in FAMILY_SPECS.values():
            self.assertEqual(len(family_spec.result_files), len(set(family_spec.result_files)))

    def test_plot_family_order_uses_family_specs(self):
        self.assertEqual(tuple(spec.key for spec in get_plot_family_specs()),
                         ('logmaps', 'tentmaps', 'lorenzs'))

    def test_kpca_remains_limited_to_logmaps_and_tentmaps(self):
        logmap_files = get_family_spec('logmaps').result_files
        tentmap_files = get_family_spec('tentmaps').result_files
        lorenz_files = get_family_spec('lorenzs').result_files

        self.assertIn('kpca_res.csv', logmap_files)
        self.assertIn('kpca_res.csv', tentmap_files)
        self.assertNotIn('kpca_res.csv', lorenz_files)

    def test_execution_steps_include_combine_stage(self):
        logmap_steps = get_execution_steps('logmaps')
        self.assertEqual(logmap_steps[-1].label, 'Combine')
        self.assertEqual(logmap_steps[-1].module, 'scripts.resgen.logmaps.Z_combine_final_res')

    def test_execution_stage_selection(self):
        self.assertEqual(len(get_execution_steps('logmaps', stage='methods')), 11)
        self.assertEqual(len(get_execution_steps('tentmaps', stage='methods')), 11)
        self.assertEqual(len(get_execution_steps('lorenzs', stage='methods')), 9)
        self.assertEqual(len(get_execution_steps('lorenzs', stage='combine')), 1)