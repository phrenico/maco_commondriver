import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from cdriver.savers.saver import save_results
from scripts.config_runall import figures_root
from scripts.experiments.combine_utils import combine_result_files
from scripts.experiments.example_logmap import Z_combine_final_res as example_logmap_combine
from scripts.experiments.noise_length import Z_combine_final_res as noise_length_combine
from scripts.experiments.lorenz_hypertune.htune_config import get_data_path_template, get_htune_paths
from scripts.plots import plot_comparisons as comparison_plots
from scripts.plots.logmaps import comparison_plot_logmap
from scripts.plots.lorenz import comparison_plot_lorenz
from scripts.plots.tentmaps import comparison_plot_tentmap


class TestExecutionPaths(unittest.TestCase):
    def test_run_all_dry_run_propagates_config(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'custom_config.py'
            config_path.write_text(
                "CONFIG_COMPARISON_PLOTS = {\n"
                "    'paths': {\n"
                "        'logmaps_final_res_path': 'paper_artifacts/results/final',\n"
                "        'tentmaps_final_res_path': 'paper_artifacts/results/final',\n"
                "        'lorenz_final_res_path': 'paper_artifacts/results/final',\n"
                "        'figure_path': 'paper_artifacts/figures',\n"
                "    },\n"
                "}\n",
                encoding='utf-8',
            )

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
        self.assertIn('scripts.plots.plot_comparisons', output)

    def test_run_all_dry_run_uses_default_config(self):
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, '-m', 'scripts.run_all', '--dry-run'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

        output = completed.stdout
        resolved_config = str((repo_root / 'scripts' / 'config_runall.py').resolve())
        self.assertIn('Starting full pipeline run...', output)
        self.assertIn('Pipeline completed successfully.', output)
        self.assertIn(f'--config {resolved_config}', output)

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

    def test_shared_comparison_plot_uses_configured_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            logmaps_final_res_path = tmp_path / 'logmaps'
            tentmaps_final_res_path = tmp_path / 'tentmaps'
            lorenz_final_res_path = tmp_path / 'lorenz'
            figure_path = tmp_path / 'figures'

            logmaps_final_res_path.mkdir()
            tentmaps_final_res_path.mkdir()
            lorenz_final_res_path.mkdir()

            pd.DataFrame(
                {
                    'method': ['PCA', 'MaCo'],
                    'r': [0.5, 0.8],
                }
            ).to_csv(logmaps_final_res_path / 'logmaps_res.csv', index=False)
            pd.DataFrame(
                {
                    'method': ['PCA', 'MaCo'],
                    'r': [0.4, 0.7],
                }
            ).to_csv(tentmaps_final_res_path / 'tentmaps_res.csv', index=False)
            pd.DataFrame(
                {
                    'method': ['PCA', 'MaCo'],
                    'r': [0.45, 0.75],
                }
            ).to_csv(lorenz_final_res_path / 'lorenz_res.csv', index=False)

            config_path = tmp_path / 'config_comparison_plots.py'
            config_path.write_text(
                "CONFIG_COMPARISON_PLOTS = {\n"
                "    'paths': {\n"
                f"        'logmaps_final_res_path': {str(logmaps_final_res_path)!r},\n"
                f"        'tentmaps_final_res_path': {str(tentmaps_final_res_path)!r},\n"
                f"        'lorenz_final_res_path': {str(lorenz_final_res_path)!r},\n"
                f"        'figure_path': {str(figure_path)!r},\n"
                "    },\n"
                "}\n",
                encoding='utf-8',
            )

            try:
                comparison_plots.main(config_path=str(config_path))
                self.assertTrue((figure_path / 'comparisons_res.png').is_file())
            finally:
                plt.close('all')

    def test_shared_comparison_plot_uses_default_config_if_args_empty(self):
        with patch('scripts.plots.plot_comparisons._get_comparison_plot_paths') as mock_get_paths:
            mock_get_paths.side_effect = Exception('Paths loaded successfully')
            with self.assertRaisesRegex(Exception, 'Paths loaded successfully'):
                comparison_plots.main(args=[])
            
            mock_get_paths.assert_called_once_with('scripts/config_runall.py')

    def test_shared_comparison_plot_requires_config_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config_missing_block.py'
            config_path.write_text('CONFIG_LOGMAPS = {}\n', encoding='utf-8')

            with self.assertRaises(AttributeError):
                comparison_plots.main(config_path=str(config_path))

    def test_shared_comparison_plot_requires_all_path_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / 'config_missing_key.py'
            config_path.write_text(
                "CONFIG_COMPARISON_PLOTS = {\n"
                "    'paths': {\n"
                f"        'logmaps_final_res_path': {str(tmp_path / 'logmaps')!r},\n"
                f"        'tentmaps_final_res_path': {str(tmp_path / 'tentmaps')!r},\n"
                f"        'figure_path': {str(tmp_path / 'figures')!r},\n"
                "    },\n"
                "}\n",
                encoding='utf-8',
            )

            with self.assertRaises(KeyError):
                comparison_plots.main(config_path=str(config_path))

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

    def test_noise_length_combine_uses_configured_plot_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            final_res_path = tmp_path / 'results'
            figure_path = tmp_path / 'figures'
            final_res_path.mkdir()

            for filename in noise_length_combine.REQUIRED_FILES:
                (final_res_path / filename).touch()

            config_path = tmp_path / 'config_noise_length.py'
            config_path.write_text(
                "CONFIG_NOISE_LENGTH = {\n"
                f"    'paths': {{'final_res_path': {str(final_res_path)!r}, 'figure_path': {str(figure_path)!r}}},\n"
                "}\n",
                encoding='utf-8',
            )

            with patch.object(sys, 'argv', ['prog', '--config', str(config_path)]):
                with patch('scripts.experiments.noise_length.Z_combine_final_res.plot_noise_length.plot_all_nl') as plot_mock:
                    noise_length_combine.main()

            plot_mock.assert_called_once_with(res_path=final_res_path, figure_path=figure_path)

    def test_noise_length_combine_falls_back_to_default_figure_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            final_res_path = tmp_path / 'results'
            final_res_path.mkdir()

            for filename in noise_length_combine.REQUIRED_FILES:
                (final_res_path / filename).touch()

            config_path = tmp_path / 'config_noise_length.py'
            config_path.write_text(
                "CONFIG_NOISE_LENGTH = {\n"
                f"    'paths': {{'final_res_path': {str(final_res_path)!r}}},\n"
                "}\n",
                encoding='utf-8',
            )

            with patch.object(sys, 'argv', ['prog', '--config', str(config_path)]):
                with patch('scripts.experiments.noise_length.Z_combine_final_res.plot_noise_length.plot_all_nl') as plot_mock:
                    noise_length_combine.main()

            plot_mock.assert_called_once_with(res_path=final_res_path, figure_path=figures_root)

    def test_logmaps_plot_uses_configured_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            final_res_path = tmp_path / 'results'
            figure_path = tmp_path / 'figures'
            final_res_path.mkdir()

            pd.DataFrame(
                {
                    'method': ['PCA', 'MaCo'],
                    'r': [0.5, 0.8],
                },
                index=[0, 1],
            ).to_csv(final_res_path / 'logmaps_res.csv')

            config_path = tmp_path / 'config_logmaps.py'
            config_path.write_text(
                "import numpy as np\n\n"
                "CONFIG_LOGMAPS = {\n"
                "    'datagen': {\n"
                "        'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),\n"
                "        'A': np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]], dtype=float),\n"
                "    },\n"
                f"    'paths': {{'final_res_path': {str(final_res_path)!r}, 'figure_path': {str(figure_path)!r}}},\n"
                "}\n",
                encoding='utf-8',
            )

            try:
                comparison_plot_logmap.main(config_path=str(config_path))
                self.assertTrue((figure_path / 'comparison_logmap.png').is_file())
            finally:
                plt.close('all')

    def test_tentmaps_plot_uses_configured_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            final_res_path = tmp_path / 'results'
            figure_path = tmp_path / 'figures'
            final_res_path.mkdir()

            pd.DataFrame(
                {
                    'method': ['PCA', 'MaCo'],
                    'r': [0.4, 0.7],
                },
                index=[0, 1],
            ).to_csv(final_res_path / 'tentmaps_res.csv')

            config_path = tmp_path / 'config_tentmaps.py'
            config_path.write_text(
                "import numpy as np\n\n"
                "CONFIG_TENTMAPS = {\n"
                "    'datagen': {\n"
                "        'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),\n"
                "    },\n"
                f"    'paths': {{'final_res_path': {str(final_res_path)!r}, 'figure_path': {str(figure_path)!r}}},\n"
                "}\n",
                encoding='utf-8',
            )

            try:
                comparison_plot_tentmap.main(config_path=str(config_path))
                self.assertTrue((figure_path / 'comparison_tentmap.png').is_file())
            finally:
                plt.close('all')

    def test_lorenz_plot_uses_configured_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            final_res_path = tmp_path / 'results'
            figure_path = tmp_path / 'figures'
            final_res_path.mkdir()

            pd.DataFrame(
                {
                    'method': ['PCA', 'MaCo'],
                    'r': [0.45, 0.75],
                }
            ).to_csv(final_res_path / 'lorenz_res.csv', index=False)

            config_path = tmp_path / 'config_lorenz.py'
            config_path.write_text(
                "CONFIG_LORENZ = {\n"
                f"    'paths': {{'final_res_path': {str(final_res_path)!r}, 'figure_path': {str(figure_path)!r}}},\n"
                "}\n",
                encoding='utf-8',
            )

            try:
                comparison_plot_lorenz.main(config_path=str(config_path))
                self.assertTrue((figure_path / 'comparisons_lorenz.png').is_file())
            finally:
                plt.close('all')

    def test_lorenz_htune_final_stage_uses_configured_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            interim_res_path = tmp_path / 'interim'
            final_res_path = tmp_path / 'final'
            figure_path = tmp_path / 'figures'
            interim_res_path.mkdir()

            for filename, method in (
                ('pca_htune.csv', 'PCA'),
                ('ica_htune.csv', 'ICA'),
                ('kpca_htune.csv', 'kPCA'),
                ('dca_htune.csv', 'DCA'),
                ('sfa_htune.csv', 'SFA'),
            ):
                pd.DataFrame(
                    {
                        'coefs': [0.8],
                        'wcomp': [1],
                        'n_components': [2],
                        'method': [method],
                        'dataset': ['lorenz'],
                    }
                ).to_csv(interim_res_path / filename)

            config_path = tmp_path / 'config_lorenz_htune.py'
            config_path.write_text(
                "CONFIG_LORENZ_HTUNE = {\n"
                f"    'paths': {{'interim_res_path': {str(interim_res_path)!r}, 'final_res_path': {str(final_res_path)!r}, 'figure_path': {str(figure_path)!r}}},\n"
                "}\n",
                encoding='utf-8',
            )

            try:
                with patch.object(sys, 'argv', ['prog', '--config', str(config_path)]):
                    runpy.run_module('scripts.experiments.lorenz_hypertune.genres_final_htune', run_name='__main__')

                self.assertTrue((final_res_path / 'htune.csv').is_file())
                self.assertTrue((figure_path / 'htune.png').is_file())
            finally:
                plt.close('all')
