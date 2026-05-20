"""Utilities for loading experiment configuration dicts.

Usage in an experiment script:
    import argparse
    from pathlib import Path
    from scripts.experiments.config_loader import get_config, resolve_paths

    _REPO_ROOT = Path(__file__).resolve().parents[3]

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default=None,
                            help='Path to external config file.  The file must '
                                 'define CONFIG_<FAMILY_KEY> (e.g. CONFIG_LOGMAPS).')
        args = parser.parse_args()

        cfg = resolve_paths(get_config('logmaps', args.config), _REPO_ROOT)
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path


def _load_module(path: str) -> types.ModuleType:
    """Dynamically load a Python file as a module."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f'Config file not found: {p}')
    spec = importlib.util.spec_from_file_location('_ext_config', str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_config(family_key: str, external_config_path: str | None = None) -> dict:
    """Return the config dict for *family_key*.

    Look-up order:
    1. If *external_config_path* is given, load that file and look for
       ``CONFIG_<FAMILY_KEY>`` (upper-cased).  If found, return it.
       If the file does not exist or does not define the attribute, raise immediately.
    2. Otherwise fall back to the built-in ``scripts.experiments.config`` module.

    Parameters
    ----------
    family_key:
        Experiment family name, e.g. ``'logmaps'``, ``'lorenz'``.
    external_config_path:
        Absolute or relative path to a ``.py`` file that may define
        ``CONFIG_<FAMILY_KEY>``.  Pass ``None`` to use the built-in defaults.

    Returns
    -------
    dict
        The configuration dict for the requested family.

    Raises
    ------
    FileNotFoundError
        If *external_config_path* is given but the file does not exist.
    AttributeError
        If *external_config_path* is given but does not define ``CONFIG_<FAMILY_KEY>``,
        or if no *external_config_path* is given and the built-in module does not
        define it either.
    """
    attr = f'CONFIG_{family_key.upper()}'

    if external_config_path is not None:
        ext_mod = _load_module(external_config_path)  # raises FileNotFoundError if missing
        if not hasattr(ext_mod, attr):
            raise AttributeError(
                f'{attr} not found in {external_config_path}. '
                f'Add a {attr} dict to that file or drop --config.'
            )
        return getattr(ext_mod, attr)

    # Built-in default (now canonical: scripts.config_runall)
    import scripts.config_runall as _default_config  # noqa: PLC0415
    if not hasattr(_default_config, attr):
        raise AttributeError(
            f'No {attr} defined in scripts.config_runall. '
            f'Available configs: '
            + ', '.join(k for k in dir(_default_config) if k.startswith('CONFIG_'))
        )
    return getattr(_default_config, attr)


def resolve_paths(config: dict, repo_root: str | Path) -> dict:
    """Return a shallow copy of *config* with ``config['paths']`` values
    resolved to absolute :class:`pathlib.Path` objects relative to *repo_root*.

    Only the top-level ``'paths'`` key is processed; all other keys are
    left unchanged (same object references).

    Parameters
    ----------
    config:
        Config dict as returned by :func:`get_config`.
    repo_root:
        Absolute path to the repository root.
    """
    repo_root = Path(repo_root)
    out = dict(config)
    if 'paths' in config:
        out['paths'] = {k: repo_root / v for k, v in config['paths'].items()}
    return out
