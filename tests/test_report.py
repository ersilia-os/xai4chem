"""The model report: stylia figures as PNG and PDF, embedded in one page, with machine-readable metrics."""

import json
import re

import pandas as pd
import pytest

from xai4chem import api
from xai4chem.report import figures as F


@pytest.fixture(scope="module")
def reported(workspace, tmp_path_factory):
    root, _ = workspace
    base = tmp_path_factory.mktemp("report")
    lib, out = base / "library", base / "m"
    api.setup(root / "library_input.csv", ["morgan", "accfg", "datamol"], lib)
    api.fit(root / "soft.csv", out, root / "hard.csv", library_dir=lib, n_estimators=150, panel_rows=300)
    return out


def test_every_figure_is_written_as_png_and_pdf_and_embedded(reported):
    figures = reported / "figures"
    pngs = sorted((figures / "png").glob("*.png"))
    assert {p.stem.split("_", 1)[1] for p in pngs} >= {
        "correlation",
        "residuals",
        "residual_structure",
        "calibration",
        "importance",
        "effects",
    }
    assert {p.stem for p in pngs} == {p.stem for p in (figures / "pdf").glob("*.pdf")}
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs)
    assert all(p.read_bytes()[:5] == b"%PDF-" for p in (figures / "pdf").glob("*.pdf"))
    text = (reported / "report.html").read_text()
    assert len(re.findall(r'<img src="data:image/png;base64,', text)) == len(pngs)  # one page, nothing linked out
    assert 'src="figures/' not in text and 'href="figures/pdf/c0_correlation.pdf"' in text


def test_hard_head_figures(reported):
    names = {p.stem for p in (reported / "figures" / "png").glob("c0_*.png")}
    assert {"c0_hard_roc", "c0_isotonic", "c0_tanimoto", "c0_hard_importance", "c0_hard_effects"} <= names


def test_metrics_are_written_next_to_the_page(reported):
    metrics = json.loads((reported / "metrics.json").read_text())
    col = metrics["columns"]["activity"]
    assert {"n", "r2", "pearson", "spearman", "rmse", "mae", "top_decile_rmse"} <= set(col) and col["a_max"] >= 0
    table = pd.read_csv(reported / "performance_table.csv")
    assert list(table["column"]) == ["activity", "noisy"]
    assert table.loc[0, "r2"] == pytest.approx(col["r2"]) and "hard_auroc" in table.columns


def test_a_rewritten_report_drops_stale_figures(reported):
    from xai4chem.report.model_report import write_report

    stale = reported / "figures" / "png" / "old.png"
    stale.write_bytes(b"x")
    write_report(reported)
    assert not stale.exists() and (reported / "figures" / "png" / "c0_correlation.png").exists()


def test_every_figure_has_a_title_and_caption():
    assert all(title and text for title, text in F.FIGURES.values())
    assert F.caption("nope") == ("nope", "")


def test_importing_stylia_keeps_matplotlibs_font_cache(tmp_path):
    """stylia deletes matplotlib's cache directory at import; a report must not pay for rebuilding the fonts."""
    import os
    import subprocess
    import sys

    marker = tmp_path / "fontlist-keep.json"
    marker.write_text("{}")
    code = "from xai4chem.report.style import import_stylia; import_stylia()"
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path)}
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
    assert marker.exists()
