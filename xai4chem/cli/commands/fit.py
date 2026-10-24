import rich_click as click

from xai4chem.cli import options
from xai4chem.cli.commands._training import run_soft, soft_options


@click.command("fit")
@soft_options
@click.option(
    "--hard-labels",
    "-h",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional measurements: SMILES plus columns named like the soft-label columns.",
)
@options.smiles_column("--hard-smiles-column")
@click.option(
    "--hard-label-columns", default=None, help="Comma-separated measurement columns (default: every numeric column)."
)
@click.option("--report/--no-report", default=True, show_default=True, help="Write the HTML model report at the end.")
@click.option("--force", is_flag=True, help="Overwrite an existing model directory.")
@options.friendly_errors
def fit_cmd(hard_labels, hard_smiles_column, hard_label_columns, report, force, **kwargs):
    """Run the whole pipeline: learn-soft → (learn-hard) → explain-global → report.

    The model directory holds everything `explain`, `predict` and `report` need.
    """
    import shutil
    import time
    from pathlib import Path

    from xai4chem.console import STEP_COLORS, echo, elapsed, rule, summary_panel
    from xai4chem.console import path as cpath

    started = time.time()
    model_dir = Path(kwargs["model_dir"])
    if (model_dir / "manifest.json").exists():
        if not force:
            raise click.ClickException(f"{model_dir} already holds a model. Pass --force to overwrite it.")
        shutil.rmtree(model_dir)
    rule("xai4chem · fit", style=STEP_COLORS["fit"], right=cpath(model_dir))
    pipeline = (
        ["learn-soft"] + (["learn-hard"] if hard_labels else []) + ["explain-global"] + (["report"] if report else [])
    )
    echo(" → ".join(pipeline), "run")
    run_soft(**kwargs)
    if hard_labels:
        from xai4chem.hard import learn_hard

        learn_hard(
            hard_labels,
            model_dir,
            library_dir=kwargs["library_dir"],
            seed=kwargs["seed"],
            smiles_col=hard_smiles_column,
            label_cols=options.split(hard_label_columns),
        )
    from xai4chem.reporting.global_explain import explain_global

    explain_global(model_dir, library_dir=kwargs["library_dir"], seed=kwargs["seed"])
    rows = [("Pipeline", " → ".join(pipeline)), ("Model", f"[dim]{cpath(model_dir)}[/]")]
    if report:
        from xai4chem.reporting.report import write_report

        out = write_report(model_dir, model_dir / "report.html")
        rows.append(("Report", f"[dim]{cpath(out)}[/]"))
    rows.append(("Next", f"xai4chem explain -m {cpath(model_dir)} -i molecules.csv -o explanations/"))
    rows.append(("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"))
    summary_panel("xai4chem · fit", rows, border_style=STEP_COLORS["fit"], icon="✓")
