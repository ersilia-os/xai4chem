"""Training options shared by `fit` and `learn-soft` (the same flags, declared once)."""

import rich_click as click

from xai4chem.cli import options


def soft_options(func):
    decorators = [
        click.option(
            "--soft-labels",
            "-s",
            required=True,
            type=click.Path(exists=True, dir_okay=False),
            help="Teacher outputs: SMILES plus one numeric column per output (CSV, TSV or Parquet).",
        ),
        options.model_dir("Model directory to create."),
        click.option(
            "--featurizer",
            "-f",
            default="morgan",
            show_default=True,
            type=click.Choice(options.FEATURIZERS),
            help="Features shared by S and H, and the vocabulary of the explanations.",
        ),
        options.library_dir(),
        click.option(
            "--transform",
            default="auto",
            show_default=True,
            type=click.Choice(["auto", "logit", "identity"]),
            help="Scale S is learned on; auto uses logit for probabilities.",
        ),
        click.option("--max-rows", default=None, type=int, help="Subsample the soft labels (quick runs)."),
        click.option(
            "--n-estimators",
            default=2000,
            show_default=True,
            type=int,
            help="Maximum boosting rounds; early stopping decides.",
        ),
        click.option(
            "--val-frac",
            default=0.1,
            show_default=True,
            type=float,
            help="Share of the soft labels held out for early stopping and fidelity.",
        ),
        click.option(
            "--stability/--no-stability",
            default=False,
            show_default=True,
            help="Train a second seed and report how stable the global importances are.",
        ),
        click.option(
            "--tune/--no-tune",
            default=False,
            show_default=True,
            help="Search boosting parameters (Optuna, on a 100k subsample) before the full fit.",
        ),
        click.option("--trials", default=20, show_default=True, type=int, help="Tuning trials (with --tune)."),
        options.seed(),
        options.smiles_column("--soft-smiles-column"),
        click.option(
            "--soft-label-columns",
            default=None,
            help="Comma-separated teacher columns (default: every numeric column).",
        ),
    ]
    for d in reversed(decorators):
        func = d(func)
    return func


def run_soft(
    soft_labels,
    model_dir,
    featurizer,
    library_dir,
    transform,
    max_rows,
    n_estimators,
    val_frac,
    stability,
    tune,
    trials,
    seed,
    soft_smiles_column,
    soft_label_columns,
):
    from xai4chem.soft import learn_soft

    return learn_soft(
        soft_labels,
        model_dir,
        featurizer_name=featurizer,
        library_dir=library_dir,
        transform=transform,
        max_rows=max_rows,
        n_estimators=n_estimators,
        val_frac=val_frac,
        stability=stability,
        tune=tune,
        trials=trials,
        seed=seed,
        smiles_col=soft_smiles_column,
        label_cols=options.split(soft_label_columns),
    )
