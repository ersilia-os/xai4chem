import rich_click as click

from xai4chem.cli import options
from xai4chem.cli.commands._training import run_soft, soft_options


@click.command("learn-soft")
@soft_options
@options.friendly_errors
def learn_soft_cmd(**kwargs):
    """Learn S, the surrogate of the teacher, from soft labels over the reference library.

    One XGBoost regressor per teacher column, on sparse features, with early stopping on held-out library
    molecules. Reports fidelity (R², Spearman) to the teacher on those molecules.
    """
    run_soft(**kwargs)
