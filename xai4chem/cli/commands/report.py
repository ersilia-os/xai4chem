import rich_click as click

from xai4chem.cli import options


@click.command("report")
@options.model_dir()
@click.option("--output", "-o", default=None, type=click.Path(), help="HTML file or directory (default: <model-dir>/report.html).")
@click.option("--open", "open_browser", is_flag=True, help="Open the report in a browser.")
@options.friendly_errors
def report_cmd(model_dir, output, open_browser):
    """Write a self-contained HTML report: fidelity, what the teacher has learned, and the hard head."""
    from pathlib import Path

    from xai4chem.reporting.report import write_report

    write_report(model_dir, output or Path(model_dir) / "report.html", open_browser=open_browser)
