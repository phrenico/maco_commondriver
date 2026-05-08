import argparse
import shlex
import subprocess
from pathlib import Path

from scripts.resgen.experiment_registry import FAMILY_SPECS, get_execution_steps, get_family_spec


def build_command(step):
    return ['conda', 'run', '-n', step.env_name, step.python_cmd, '-m', step.module]


def format_command(command):
    return ' '.join(shlex.quote(part) for part in command)


def run_step(step, repo_root, dry_run=False):
    command = build_command(step)
    print(f'[{step.label}] {format_command(command)}')
    if not dry_run:
        subprocess.run(command, check=True, cwd=repo_root)


def main():
    parser = argparse.ArgumentParser(description='Run a canonical experiment family pipeline.')
    parser.add_argument('family', choices=tuple(FAMILY_SPECS))
    parser.add_argument('--stage', choices=('all', 'methods', 'combine'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    family_spec = get_family_spec(args.family)
    steps = get_execution_steps(args.family, stage=args.stage)

    print(f'Running family: {family_spec.title}')
    print(f'Stage: {args.stage}')
    print(f'Dry run: {args.dry_run}')

    total_steps = len(steps)
    for index, step in enumerate(steps, start=1):
        print(f'[{index}/{total_steps}] {step.label}')
        run_step(step, repo_root, dry_run=args.dry_run)


if __name__ == '__main__':
    main()