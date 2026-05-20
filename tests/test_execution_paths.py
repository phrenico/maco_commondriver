import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cdriver.savers.saver import save_results
from scripts.config_runall import figures_root
from scripts.experiments.combine_utils import combine_result_files
from scripts.experiments.example_logmap import Z_combine_final_res as example_logmap_combine
from scripts.experiments.lorenz_hypertune.htune_config import get_data_path_template, get_htune_paths


class TestExecutionPaths(unittest.TestCase):
    def test_run_all_dry_run_propagates_config(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'custom_config.py'
            config_path.write_text('# dry-run placeholder\n', encoding='utf-8')

            completed = subprocess.run(
                [sys.executable, '-m', 'scripts.run_all', '--dry-run', '--config', str(config_path)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

        output = completed.stdout
        resolved_config = str(config_path.resolve())
        self.assertIn('Starting full pipeline run...', output)
        self.assertIn('Pipeline completed successfully.', output)
        self.assertIn(f'--config {resolved_config}', output)
        self.assertIn('scripts.experiments.run_family logmaps', output)

    def test_save_results_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'nested' / 'results' / 'random_res.csv'

            df = save_results(
                fname=output_path,
                r=[0.1, 0.2],
                N=2,
                method='Random',
                dataset='logmap',
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(df['method'].tolist(), ['Random', 'Random'])

    def test_combine_result_files_creates_final_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            interim_path = tmp_path / 'interim'
            interim_path.mkdir()

            pd.DataFrame({'r': [0.1], 'method': ['PCA']}, index=[0]).to_csv(interim_path / 'a.csv')
            pd.DataFrame({'r': [0.2], 'method': ['ICA']}, index=[1]).to_csv(interim_path / 'b.csv')

            final_csv_path = tmp_path / 'final' / 'combined' / 'results.csv'
            combined = combine_result_files(interim_path, final_csv_path, ('a.csv', 'b.csv'))

            self.assertTrue(final_csv_path.is_file())
            self.assertEqual(combined['method'].tolist(), ['PCA', 'ICA'])

    def test_lorenz_htune_helpers_use_configured_paths(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = {
                'paths': {
                    'interim_res_path': tmp_path / 'interim',
                    'final_res_path': tmp_path / 'final',
                    'figure_path': tmp_path / 'figures',
                },
                'data': {
                    'data_path_template': 'data/lorenz/lorenz_{}.npz',
                    'N': 1,
                },
                'preprocessing': {
                    'train_split': 0.5,
                    'valid_split': 0.25,
                },
            }

            paths = get_htune_paths(cfg)

            self.assertTrue(paths['interim_res_path'].is_dir())
            self.assertTrue(paths['final_res_path'].is_dir())
            self.assertTrue(paths['figure_path'].is_dir())
            self.assertEqual(
                get_data_path_template(cfg, repo_root),
                str(repo_root / 'data/lorenz/lorenz_{}.npz'),
            )

    def test_example_logmap_combine_uses_configured_plot_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            final_res_path = tmp_path / 'results'
            figure_path = tmp_path / 'figures'
            final_res_path.mkdir()

            for filename in example_logmap_combine.REQUIRED_FILES:
                (final_res_path / filename).touch()

            config_path = tmp_path / 'config_example.py'
            config_path.write_text(
                "CONFIG_EXAMPLE_LOGMAP = {\n"
                f"    'paths': {{'final_res_path': {str(final_res_path)!r}, 'figure_path': {str(figure_path)!r}}},\n"
                "}\n",
                encoding='utf-8',
            )

            with patch.object(sys, 'argv', ['prog', '--config', str(config_path)]):
                with patch('scripts.experiments.example_logmap.Z_combine_final_res.plot_example_logmap_res') as plot_mock:
                    example_logmap_combine.main()

            plot_mock.assert_called_once_with(res_path=final_res_path, figure_path=figure_path)

    def test_example_logmap_combine_falls_back_to_default_figure_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            final_res_path = tmp_path / 'results'
            final_res_path.mkdir()

            for filename in example_logmap_combine.REQUIRED_FILES:
                (final_res_path / filename).touch()

            config_path = tmp_path / 'config_example.py'
            config_path.write_text(
                "CONFIG_EXAMPLE_LOGMAP = {\n"
                f"    'paths': {{'final_res_path': {str(final_res_path)!r}}},\n"
                "}\n",
                encoding='utf-8',
            )

            with patch.object(sys, 'argv', ['prog', '--config', str(config_path)]):
                with patch('scripts.experiments.example_logmap.Z_combine_final_res.plot_example_logmap_res') as plot_mock:
                    example_logmap_combine.main()

            plot_mock.assert_called_once_with(res_path=final_res_path, figure_path=figures_root)
