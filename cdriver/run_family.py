"""run_family – train an ensemble (family) of MaCo models.

This module is the canonical public entry-point for the MaCo training pipeline.
Install the package (``pip install -e .``) and run::

    cdriver-run --help

or call :func:`main` directly from Python.
"""

import argparse
import pickle
import numpy as np
import torch

from .data import load_data
from .model import MaCo


# ---------------------------------------------------------------------------
# Command-construction helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Train a family of MaCo models on the sample data.",
    )
    p.add_argument(
        "--data",
        default="data/sampledata.csv",
        metavar="PATH",
        help="Path to the input CSV file (default: data/sampledata.csv).",
    )
    p.add_argument(
        "--out-dir",
        default="scripts_and_results/resdata",
        metavar="DIR",
        help="Directory for output artefacts (default: scripts_and_results/resdata).",
    )
    p.add_argument(
        "--n-models",
        type=int,
        default=50,
        metavar="N",
        help="Number of models in the family (default: 50).",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=4000,
        metavar="N",
        help="Training epochs per model (default: 4000).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        metavar="N",
        help="Mini-batch size (default: 2000).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        metavar="LR",
        help="Adam learning-rate (default: 0.01).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducibility (default: None).",
    )
    return p


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------

def run_family(
    csv_path: str,
    out_dir: str,
    n_models: int = 50,
    n_epochs: int = 4000,
    batch_size: int = 2000,
    lr: float = 1e-2,
    seed: int | None = None,
) -> dict:
    """Train *n_models* MaCo models and save artefacts to *out_dir*.

    Returns a summary dict with ``'r_predict'``, ``'r_reconst'``, and
    ``'ind_best_model'`` keys so callers can inspect results without reading
    files.
    """
    import os
    import pandas as pd
    from tqdm import tqdm

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model hyper-parameters
    dx, dy, dz, nh = 1, 2, 1, 20
    mapper_kwargs = dict(n_h1=nh, n_h2=nh)
    coach_kwargs = dict(n_h1=nh)

    # Data
    train_loader, test_loader, valid_loader, z_valid = load_data(
        csv_path,
        batch_size=batch_size,
    )

    models = [
        MaCo(Ex=dx, Ey=dy, Ez=dz,
             mh_kwargs=mapper_kwargs, ch_kwargs=coach_kwargs, device=device)
        for _ in range(n_models)
    ]

    # Train
    train_losses, test_loss = [], []
    for model in tqdm(models, desc="Training models"):
        train_losses.append(model.train_loop(train_loader, n_epochs, lr=lr))
        test_loss.append(model.test_loop(test_loader))
    train_losses = np.array(train_losses).T

    # Pick best model
    ind_best = int(np.argmin(test_loss))
    best_model = models[ind_best]

    # Validation
    _val_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(valid_loader)

    # Per-model correlation metrics
    r_predict, r_reconst = [], []
    for model in models:
        _, xp, zp, _ = model.valid_loop(valid_loader)
        r_predict.append(np.corrcoef(xp, valid_loader[1][:, 0].numpy())[0, 1])
        r_reconst.append(np.corrcoef(zp, z_valid.squeeze().numpy())[0, 1])

    # Persist artefacts
    res = dict(
        cc_pred=z_pred,
        cc_valid=z_valid.squeeze().numpy(),
        x_valid=valid_loader[1].squeeze().detach().numpy(),
        x_past_valid=valid_loader[0][:, 0].numpy(),
        x_pred=x_pred,
        Y_1_valid=valid_loader[0][:, 1].numpy(),
        Y_2_valid=valid_loader[0][:, 2].numpy(),
    )
    pd.DataFrame(res).to_csv(os.path.join(out_dir, "mappercoach_res.csv"))
    np.save(os.path.join(out_dir, "learning_curves.npy"), train_losses)
    np.save(os.path.join(out_dir, "test_loss.npy"), test_loss)
    torch.save(best_model, os.path.join(out_dir, "best_model.pth"))
    with open(os.path.join(out_dir, "models.pkl"), "wb") as fh:
        pickle.dump(models, fh)
    pd.DataFrame({"r_predict": r_predict, "r_reconst": r_reconst}).to_csv(
        os.path.join(out_dir, "r_values.csv")
    )

    return dict(r_predict=r_predict, r_reconst=r_reconst, ind_best_model=ind_best)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """Parse *argv* (or ``sys.argv``) and call :func:`run_family`."""
    args = _build_parser().parse_args(argv)
    summary = run_family(
        csv_path=args.data,
        out_dir=args.out_dir,
        n_models=args.n_models,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"Best model index: {summary['ind_best_model']}")
    r2_pred = np.mean(np.array(summary["r_predict"]) ** 2)
    r2_rec = np.mean(np.array(summary["r_reconst"]) ** 2)
    print(f"Mean R² prediction:     {r2_pred:.4f}")
    print(f"Mean R² reconstruction: {r2_rec:.4f}")


if __name__ == "__main__":
    main()
