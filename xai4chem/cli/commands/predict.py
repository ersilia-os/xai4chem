import rich_click as click

from xai4chem.cli import options


@click.command("predict")
@options.model_dir()
@click.option("--input", "-i", "input_file", required=True, type=click.Path(exists=True, dir_okay=False), help="Molecules to score: a CSV/TSV/Parquet with a SMILES column.")
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False), help="Output CSV.")
@options.columns()
@options.smiles_column()
@click.option("--details", is_flag=True, help="Also write S, H_S, T and a per column.")
@options.friendly_errors
def predict_cmd(model_dir, input_file, output, columns, smiles_column, details):
    """Score molecules fast, without explanations (millions of molecules are fine)."""
    import time

    from xai4chem.console import STEP_COLORS, elapsed, rule, set_active_color, summary_panel, sweep_progress
    from xai4chem.console import path as cpath
    from xai4chem.io import read_table, smiles_column as find_smiles
    from xai4chem.model import XAIModel

    started = time.time()
    color = STEP_COLORS["predict"]
    set_active_color(color)
    df = read_table(input_file)
    smiles = df[find_smiles(df, smiles_column)].astype(str).tolist()
    model = XAIModel(model_dir)
    rule("xai4chem · predict", style=color, right=f"{len(smiles):,} molecules · {cpath(model_dir)}")
    with sweep_progress("scoring", len(smiles)) as tick:
        out = model.predict(smiles, columns=options.split(columns), details=details, progress=tick)
    out.to_csv(output, index=False)
    n_bad = int(out.iloc[:, 1].isna().sum()) if out.shape[1] > 1 else 0
    rows = [("Molecules", f"{len(smiles):,}" + (f" · {n_bad:,} unparseable" if n_bad else "")), ("Output", f"[dim]{cpath(output)}[/]")]
    rows.append(("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"))
    summary_panel("xai4chem · predict", rows, border_style=color, icon="✓")
