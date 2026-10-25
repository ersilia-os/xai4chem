import rich_click as click

from xai4chem.cli import options


@click.command("predict")
@options.model_dir()
@options.molecules_input("score")
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False), help="Output CSV.")
@options.columns()
@options.smiles_column()
@click.option("--details", is_flag=True, help="Also write S, H_S, T and a per column.")
@click.option(
    "--n-jobs", "-j", default=None, type=int, help="Worker processes for featurization (default: CPUs − 1, at most 8)."
)
@options.friendly_errors
def predict_cmd(model_dir, input_file, smiles_list, output, columns, smiles_column, details, n_jobs):
    """Score molecules fast, without explanations (millions of molecules are fine)."""
    import time

    from xai4chem.console import STEP_COLORS, elapsed, rule, set_active_color, summary_panel, sweep_progress
    from xai4chem.console import path as cpath
    from xai4chem.model import XAIModel

    started = time.time()
    color = STEP_COLORS["predict"]
    set_active_color(color)
    smiles, _ = options.load_molecules(input_file, smiles_list, smiles_column)
    model = XAIModel(model_dir)
    rule("xai4chem · predict", style=color, right=f"{len(smiles):,} molecules · {cpath(model_dir)}")
    with sweep_progress("scoring", len(smiles)) as tick:
        out = model.predict(smiles, columns=options.split(columns), details=details, progress=tick, n_jobs=n_jobs)
    out.to_csv(output, index=False)
    n_bad = int((~out["valid"]).sum())
    rows = [
        ("Molecules", f"{len(smiles):,}" + (f" · {n_bad:,} unparseable" if n_bad else "")),
        ("Output", f"[dim]{cpath(output)}[/]"),
    ]
    rows.append(("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"))
    summary_panel("xai4chem · predict", rows, border_style=color, icon="✓")
