import argparse
import shlex
import subprocess
from pathlib import Path

from scripts.experiments.experiment_registry import FAMILY_SPECS, get_execution_steps, get_family_spec


def get_uv_env_dir(step, repo_root):
    env_dir = repo_root / 'envs' / step.env_name
    if not env_dir.is_dir():
        raise FileNotFoundError(
            f"UV environment directory not found for '{step.env_name}': {env_dir}"
        )

    pyproject = env_dir / 'pyproject.toml'
    if not pyproject.is_file():
        raise FileNotFoundError(
            f"Missing pyproject.toml for UV environment '{step.env_name}': {pyproject}"
        )
    return env_dir


def build_command(step, config_path=None):
    cmd = ['uv', 'run', step.python_cmd, '-m', step.module]
    if config_path is not None:
        cmd += ['--config', config_path]
    return cmd


def format_command(command):
    return ' '.join(shlex.quote(part) for part in command)


def run_step(step, repo_root, dry_run=False, config_path=None):
    env_dir = get_uv_env_dir(step, repo_root)
    command = build_command(step, config_path=config_path)
    print(f'[{step.label}] (cwd={env_dir}) {format_command(command)}')
    if not dry_run:
        subprocess.run(command, check=True, cwd=env_dir)


def main():
    parser = argparse.ArgumentParser(description='Run a canonical experiment family pipeline.')
    parser.add_argument('family', choices=tuple(FAMILY_SPECS))
    parser.add_argument('--stage', choices=('all', 'methods', 'combine'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--config', default=None,
                        help='Path to external config file (resolved to absolute path).')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    # Resolve config path to absolute immediately (scripts run with different cwd)
    config_path = None
    if args.config is not None:
        config_path = str(Path(args.config).resolve())
    family_spec = get_family_spec(args.family)
    steps = get_execution_steps(args.family, stage=args.stage)

    print(f'Running family: {family_spec.title}')
    print(f'Stage: {args.stage}')
    print(f'Dry run: {args.dry_run}')

    total_steps = len(steps)
    for index, step in enumerate(steps, start=1):
        print(f'[{index}/{total_steps}] {step.label}')
        run_step(step, repo_root, dry_run=args.dry_run, config_path=config_path)


if __name__ == '__main__':
    main()