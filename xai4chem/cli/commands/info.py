import rich_click as click

from xai4chem.cli import options


@click.command("info")
@options.model_dir()
@options.friendly_errors
def info_cmd(model_dir):
    """Describe a model: featurizer, columns, fidelity, hard head and top features."""
    from xai4chem.info import show_info

    show_info(model_dir)
