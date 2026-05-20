import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.experiments.experiment_registry import FAMILY_SPECS


def format_command(command):
	return ' '.join(shlex.quote(part) for part in command)


def run_command(command, repo_root, dry_run=False):
	print(format_command(command))
	if not dry_run:
		subprocess.run(command, check=True, cwd=repo_root)


def run_all_experiments(repo_root, dry_run=False, config_path=None):
	family_keys = tuple(FAMILY_SPECS)
	total_families = len(family_keys)

	for index, family_key in enumerate(family_keys, start=1):
		print(f'[{index}/{total_families}] Running family: {family_key}')
		command = [sys.executable, '-m', 'scripts.experiments.run_family', family_key]
		if dry_run:
			command.append('--dry-run')
		if config_path is not None:
			command += ['--config', config_path]
		run_command(command, repo_root, dry_run=dry_run)


def run_comparison_plot(repo_root, dry_run=False):
	command = [sys.executable, '-m', 'scripts.plots.plot_comparisons']
	print('Running comparison plot...')
	run_command(command, repo_root, dry_run=dry_run)


def main():
	parser = argparse.ArgumentParser(
		description='Run all registered experiment families and generate comparison plots.'
	)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--config', default=None,
						help='Path to external config file (resolved to absolute path).')
	args = parser.parse_args()

	repo_root = Path(__file__).resolve().parents[1]

	# Resolve to absolute path immediately so subprocesses can use it.
	config_path = None
	if args.config is not None:
		config_path = str(Path(args.config).resolve())

	print('Starting full pipeline run...')
	print(f'Dry run: {args.dry_run}')

	run_all_experiments(repo_root, dry_run=args.dry_run, config_path=config_path)
	if not args.dry_run:
		run_comparison_plot(repo_root, dry_run=False)

	print('Pipeline completed successfully.')


if __name__ == '__main__':
	main()
