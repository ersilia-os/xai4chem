import rich_click as click

from xai4chem.cli import options


@click.command("learn-hard")
@click.option("--hard-labels", "-h", required=True, type=click.Path(exists=True, dir_okay=False), help="Your measurements: SMILES plus columns named like the soft-label columns (0/1 → classifier).")
@options.model_dir("Model directory written by learn-soft.")
@options.library_dir()
@options.seed()
@options.smiles_column("--hard-smiles-column")
@click.option("--hard-label-columns", default=None, help="Comma-separated measurement columns (default: every numeric column).")
@options.friendly_errors
def learn_hard_cmd(hard_labels, model_dir, library_dir, seed, hard_smiles_column, hard_label_columns):
    """Learn H from hard labels, calibrate it onto S's scale, and set up the Tanimoto gate.

    A hard column matches a soft column by name, or as a prefix up to `_`/`-`
    (`abaumannii_inhibition` → `abaumannii_inhibition_probability`).
    """
    from xai4chem.hard import learn_hard

    learn_hard(hard_labels, model_dir, library_dir=library_dir, seed=seed, smiles_col=hard_smiles_column, label_cols=options.split(hard_label_columns))
