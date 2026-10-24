import rich_click as click

from xai4chem.cli import options


@click.command("validate")
@options.model_dir()
@click.option(
    "--soft-labels",
    "-s",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Held-out teacher outputs (fidelity of S and P).",
)
@click.option(
    "--hard-labels",
    "-h",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Held-out measurements (how well S and P predict them).",
)
@click.option("--output", "-o", default=None, type=click.Path(dir_okay=False), help="Write the metrics as JSON.")
@options.friendly_errors
def validate_cmd(model_dir, soft_labels, hard_labels, output):
    """Score a model on held-out soft labels and/or measurements, comparing S with the blended P."""
    if soft_labels is None and hard_labels is None:
        raise click.ClickException("Pass --soft-labels and/or --hard-labels.")
    from xai4chem.validate import validate

    validate(model_dir, soft_labels=soft_labels, hard_labels=hard_labels, output=output)
