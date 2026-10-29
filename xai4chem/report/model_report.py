"""``report.html``: fidelity, what the teacher has learned, and the hard head, as one self-contained page.

Like Olinda's and ZairaChem's reports, the figures are drawn with stylia: each one is written as a PNG and a
vector PDF under ``figures/`` and the PNG is embedded in the page, so the page stays a single file. Next to it
go ``metrics.json`` and ``performance_table.csv``, the numbers behind the page in machine-readable form.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from xai4chem.report import figures as F
from xai4chem.report import html as H
from xai4chem.report.drawing import fragment_svg
from xai4chem.run import (
    EXPLAIN_DIRNAME,
    FIDELITY_NAME,
    FIGURES_DIRNAME,
    H_HELDOUT_NAME,
    H_TO_S_NAME,
    METRICS_NAME,
    PERFORMANCE_NAME,
    REPORT_NAME,
    column_dir,
    hard_dir,
    read_json,
    read_manifest,
    write_json,
)

TOP = 20

METHODS = """
<details style="margin-top:8px"><summary>How to read this report</summary>
<div class="card" style="margin-top:10px; font-size:12.5px; line-height:1.55">
<p><b>S</b> is a gradient-boosting surrogate trained to reproduce the teacher's outputs (soft labels) over the
reference library, on the teacher's raw scale. Its <b>fidelity</b> is measured on library
molecules held out from training. Explanations describe S, so they are as trustworthy as that fidelity.</p>
<p>Attributions are exact TreeSHAP values on S's scale. <b>Importance</b> is the mean |SHAP| over the explanation
panel, a fixed random subset of the library. <b>Direction</b> is the correlation between a feature's value and its
SHAP value: positive means more of the feature raises the prediction. <b>When present</b> is the mean |SHAP| over
the molecules that contain the feature, which ranks rare but strong features fairly.</p>
__MORGAN__<p>With hard labels, <b>H</b> is trained on the measurements with the same features and carried onto S's
scale by an isotonic map (<b>H_S</b>). The prediction is P = (1 − a)·S + a·H_S, where a = a_max · ramp(T) and T is
the Tanimoto similarity (binary Morgan) to the nearest labelled molecule, so the hard head only speaks near the
labelled chemistry. Because a is fixed for a given molecule, the explanation of P is exactly (1 − a)·φ_S + a·φ_HS.</p>
</div></details>
"""

MORGAN_NOTE = """<p>Morgan bits are folded, so one bit can be set by several substructures. Each bit is named by the
substructure that most often sets it in the library, with that substructure's share; <b>collision</b> is the share of
occurrences that come from other substructures.</p>
"""


def _f(v, fmt="{:.3f}"):
    try:
        return fmt.format(v) if v is not None and v == v else "—"
    except (TypeError, ValueError):
        return "—"


def _direction_badge(d):
    if d > 0.05:
        return H.badge("raises", hue="crimson")
    if d < -0.05:
        return H.badge("lowers", hue="cobalt")
    return H.badge("mixed")


def _drawing(feature, featurizer, bit_dictionary, smarts):
    if featurizer == "datamol":
        return ""  # descriptors have no substructure to draw
    index = int(str(feature).split("-")[1])
    if featurizer == "morgan" and bit_dictionary is not None:
        frags = bit_dictionary.get(str(index), {}).get("fragments", [])
        return fragment_svg(frags[0]["smiles"], size=64) if frags else ""
    if featurizer == "accfg" and smarts is not None:
        return fragment_svg(smarts[index], size=64, smarts=True)
    return ""


def _top_table(df, featurizer, bit_dictionary=None, smarts=None):
    counts = featurizer != "datamol"  # presence only means something for count features
    vmax = float(df["mean_abs_shap"].head(TOP).max()) or 1.0
    rows = []
    for _, r in df.head(TOP).iterrows():
        drawing = _drawing(r["feature"], featurizer, bit_dictionary, smarts)
        cells = [
            f'<div class="frag">{drawing}</div>' if drawing else "",
            H.esc(r["label"]),
            f'<span class="ibar" style="width:{60 * r["mean_abs_shap"] / vmax:.0f}px"></span> {_f(r["mean_abs_shap"])}',
            _direction_badge(r["direction"]),
        ]
        if counts:
            cells += [_f(r["frequency"], "{:.1%}"), _f(r["mean_abs_shap_when_present"])]
        if "collision" in df.columns:
            cells.append(_f(r["collision"], "{:.0%}"))
        rows.append(cells)
    headers = ["", "Feature", "Importance", "Direction"] + (["Frequency", "When present"] if counts else [])
    if "collision" in df.columns:
        headers.append("Collision")
    return H.table(headers, rows, numeric=(2, 4, 5, 6) if counts else (2,))


def _substructures(table, bit_dictionary):
    """Rows whose bit is named by an environment of radius ≥ 1 (single-atom bits mostly track size)."""

    def radius(feature):
        frags = bit_dictionary.get(str(feature).split("-")[1], {}).get("fragments", [])
        return frags[0]["radius"] if frags else 0

    return table[[radius(f) >= 1 for f in table["feature"]]].reset_index(drop=True)


def _effects_data(plot_path):
    data = np.load(plot_path)
    return list(data["labels"]), data["phi"], data["values"]


def _training_details(meta):
    fid = meta["fidelity"]
    rows = [
        ["Training rows", f"{meta['n_train']:,}"],
        ["Held-out rows", f"{meta['n_val']:,}"],
        ["Trees", f"{meta['n_trees']:,}"],
        ["Fidelity", f"R² {_f(fid['r2'])} · RMSE {_f(fid['rmse'])} · MAE {_f(fid['mae'])}"],
        ["Top-decile RMSE", _f(fid["top_decile_rmse"])],
        ["Reweighting", H.esc(f"{meta['reweight']['strategy']} ({meta['reweight']['reason']})")],
    ]
    rows += [[H.esc(k), H.esc(v)] for k, v in meta["params"].items()]
    return (
        '<details style="margin-top:14px"><summary>Training details</summary>' + H.table(["", ""], rows) + "</details>"
    )


def _heldout_text(hard):
    ho = hard["heldout"]
    return f"AUROC {_f(ho['auroc'])}" if "auroc" in ho else f"ρ {_f(ho['spearman'])}"


def _performance_rows(model_dir, manifest):
    """One row of numbers per teacher column, the same ones the page, ``metrics.json`` and the CSV show."""
    rows = []
    for col in manifest["columns"]:
        cdir = column_dir(model_dir, col["id"])
        summary = read_json(cdir / EXPLAIN_DIRNAME / "summary.json")
        meta = read_json(cdir / "soft_meta.json")
        row = {"column": col["name"], **meta["fidelity"], "reweighting": meta["reweight"]["strategy"]}
        hard = summary.get("hard")
        if hard:
            row.update(
                hard_task=hard["task"],
                hard_labels=hard["n_labels"],
                hard_heldout=hard["heldout"],
                a_max=hard["a_max"],
                alignment_r2=hard["alignment_r2"],
                reach=hard["reach"],
            )
        rows.append(row)
    return rows


def _write_metrics(model_dir, manifest, rows):
    """``metrics.json`` and ``performance_table.csv``: the page's numbers, machine-readable."""
    write_json(
        Path(model_dir) / METRICS_NAME,
        {
            "xai4chem_version": manifest["xai4chem_version"],
            "featurizer": manifest["featurizer"]["name"],
            "columns": {r["column"]: {k: v for k, v in r.items() if k != "column"} for r in rows},
        },
    )
    flat = []
    for r in rows:
        r = dict(r)
        held = r.pop("hard_heldout", None) or {}
        r.update({f"hard_{k}": v for k, v in held.items()})
        flat.append(r)
    pd.DataFrame(flat).to_csv(Path(model_dir) / PERFORMANCE_NAME, index=False)


def _performance_table(rows):
    cells = []
    for r in rows:
        hard = f"{r['hard_task']} · {_heldout_text({'heldout': r['hard_heldout']})}" if "hard_task" in r else "—"
        cells.append(
            [
                H.esc(r["column"]),
                f"{r['n']:,}",
                _f(r["r2"]),
                _f(r["spearman"]),
                _f(r["rmse"]),
                _f(r["top_decile_rmse"]),
                H.esc(hard),
                _f(r["a_max"], "{:.2f}") if "a_max" in r else "—",
            ]
        )
    return H.table(
        ["Column", "Held-out", "R²", "ρ", "RMSE", "Top-decile RMSE", "Hard head", "a_max"],
        cells,
        numeric=(1, 2, 3, 4, 5, 7),
    )


def write_report(model_dir, output=None):
    """Write the model report from what ``fit`` saved in the model directory.

    Writes ``report.html``, the figures it embeds (``figures/png`` and ``figures/pdf``), ``metrics.json`` and
    ``performance_table.csv``.

    Parameters
    ----------
    model_dir : str or Path
        A model directory written by ``xai4chem fit``.
    output : str or Path, optional
        Where to write the page (default ``<model_dir>/report.html``).

    Returns
    -------
    Path
        The page written.
    """
    from xai4chem.featurizers import featurizer_from_config

    model_dir = Path(model_dir)
    output = Path(output) if output else model_dir / REPORT_NAME
    manifest = read_manifest(model_dir)
    featurizer = manifest["featurizer"]["name"]
    bit_dictionary = None
    if (model_dir / "bit_dictionary.json").exists():
        bit_dictionary = read_json(model_dir / "bit_dictionary.json")["bits"]
    smarts = featurizer_from_config(manifest["featurizer"]).smarts() if featurizer == "accfg" else None
    figures_dir = model_dir / FIGURES_DIRNAME
    shutil.rmtree(figures_dir, ignore_errors=True)  # a rewritten report must not keep a stale figure
    prefix = (
        Path(figures_dir).relative_to(output.parent).as_posix() + "/" if output.parent in figures_dir.parents else ""
    )
    rows = _performance_rows(model_dir, manifest)
    _write_metrics(model_dir, manifest, rows)
    body = [
        H.section(
            "Performance",
            _performance_table(rows),
            hint=f"held-out library molecules; the same numbers are in {METRICS_NAME} and {PERFORMANCE_NAME}",
        )
    ]
    for col in manifest["columns"]:
        exp = column_dir(model_dir, col["id"]) / EXPLAIN_DIRNAME
        hard = read_json(exp / "summary.json").get("hard")
        body.append(_soft_section(model_dir, figures_dir, prefix, col, featurizer, bit_dictionary, smarts))
        if hard:
            body.append(
                _hard_section(model_dir, figures_dir, prefix, col, hard, exp, featurizer, bit_dictionary, smarts)
            )
    lede = (
        f"Surrogate of a teacher model, learned from soft labels over the {H.esc(manifest['library']['name'])} reference "
        f"library ({manifest['library']['n_molecules']:,} molecules) with {H.esc(featurizer)} features. "
        f"xai4chem {H.esc(manifest['xai4chem_version'])}."
    )
    methods = METHODS.replace("__MORGAN__", MORGAN_NOTE if featurizer == "morgan" else "")
    output.write_text(
        H.page(
            f"xai4chem · {model_dir.name}",
            "xai4chem",
            f"{H.esc(model_dir.name)} · model <em>report</em>",
            lede,
            methods + "".join(body),
        )
    )
    return output


def _soft_section(model_dir, figures_dir, prefix, col, featurizer, bit_dictionary, smarts):
    """What the teacher has learned, for one column: fidelity, figures, top features, training details."""
    cdir = column_dir(model_dir, col["id"])
    exp = cdir / EXPLAIN_DIRNAME
    s = read_json(exp / "summary.json")
    meta = read_json(cdir / "soft_meta.json")
    fid, hard = s["fidelity"], s.get("hard")
    stats = [
        H.stat("Fidelity R²", _f(fid["r2"]), f"{fid['n']:,} held-out library molecules"),
        H.stat("Spearman ρ", _f(fid["spearman"]), "rank agreement with the teacher"),
        H.stat("Top-decile RMSE", _f(fid["top_decile_rmse"]), "error on the teacher's highest 10%"),
    ]
    if hard:
        key, value = (
            ("Held-out AUROC", hard["heldout"]["auroc"])
            if "auroc" in hard["heldout"]
            else (
                "Held-out ρ",
                hard["heldout"]["spearman"],
            )
        )
        stats.append(H.stat(key, _f(value), f"H on {hard['n_labels']:,} labels"))
        stats.append(
            H.stat("a_max", _f(hard["a_max"], "{:.2f}"), f"R² of H_S vs teacher {_f(hard['alignment_r2'], '{:.2f}')}")
        )
    top_s = pd.read_csv(exp / "global_s.csv")
    cid = col["id"]

    def draw(key, fn):
        return F.render(fn, figures_dir, f"{cid}_{key}", key)

    fidelity_figs = []
    if (cdir / FIDELITY_NAME).exists():
        pair = np.load(cdir / FIDELITY_NAME)
        t, p = pair["teacher"], pair["surrogate"]
        fidelity_figs = [
            draw("correlation", lambda ax, st: F.correlation(ax, st, t, p, fid)),
            draw("residuals", lambda ax, st: F.residuals(ax, st, t, p)),
            draw("residual_structure", lambda ax, st: F.residual_structure(ax, st, t, p)),
            draw("calibration", lambda ax, st: F.calibration(ax, st, t, p)),
        ]
    labels, phi, values = _effects_data(exp / "plot_s.npz")
    explain_figs = [
        draw(
            "importance",
            lambda ax, st: F.importance(ax, st, top_s["label"], top_s["mean_abs_shap"], top_s["direction"]),
        ),
        draw("effects", lambda ax, st: F.effects(ax, st, labels, phi, values)),
    ]
    content = [H.kpis(stats), H.figures(fidelity_figs, prefix), H.figures(explain_figs, prefix)]
    content.append(
        '<h3 style="margin-top:18px; font-size:13px">Top features</h3>'
        + _top_table(top_s, featurizer, bit_dictionary, smarts)
    )
    if featurizer == "morgan" and bit_dictionary is not None:
        subs = _substructures(top_s, bit_dictionary)
        if len(subs):
            content.append(
                '<h3 style="margin-top:18px; font-size:13px">Top substructures</h3>'
                '<p class="faint" style="font-size:11.5px;margin:0 0 6px">Bits whose named environment spans at least '
                "one bond; single-atom bits mostly track molecular size and composition.</p>"
                + _top_table(subs, featurizer, bit_dictionary, smarts)
            )
    content.append(_training_details(meta))
    return H.section(
        f"{col['name']} · what the teacher has learned",
        "".join(content),
        hint=f"panel of {s['panel_size']:,} library molecules",
    )


def _hard_section(model_dir, figures_dir, prefix, col, hard, exp, featurizer, bit_dictionary, smarts):
    reach = np.load(exp / "reach.npz")
    cid = col["id"]
    stats = [
        H.stat("Task", hard["task"], f"{hard['n_labels']:,} labelled molecules"),
        H.stat("Reach", _f(hard["reach"], "{:.1%}"), f"of panel molecules have a > 0 (T ≥ {hard['t_lo']})"),
        H.stat("Agreement", _f(hard["importance_spearman_s_vs_hs"], "{:.2f}"), "S vs H_S importances (ρ)"),
        H.stat("Calibration", hard["isotonic_direction"], "isotonic H → S map"),
    ]
    hdir = hard_dir(model_dir, cid)
    iso = read_json(hdir / H_TO_S_NAME)
    top_h = pd.read_csv(exp / "global_hs.csv")
    labels, phi, values = _effects_data(exp / "plot_hs.npz")

    def draw(key, fn):
        return F.render(fn, figures_dir, f"{cid}_{key}", key)

    held = np.load(hdir / H_HELDOUT_NAME) if (hdir / H_HELDOUT_NAME).exists() else None
    performance = None
    if held is not None and "auroc" in hard["heldout"]:
        performance = draw(
            "hard_roc", lambda ax, st: F.hard_roc(ax, st, held["y"], held["margin"], hard["heldout"]["auroc"])
        )
    elif held is not None:
        performance = draw(
            "hard_agreement",
            lambda ax, st: F.hard_agreement(ax, st, held["y"], held["margin"], hard["heldout"]["spearman"]),
        )
    gate_figs = [
        performance,
        draw("isotonic", lambda ax, st: F.isotonic(ax, st, iso["x"], iso["y"], reach["h"])),
        draw("tanimoto", lambda ax, st: F.tanimoto(ax, st, reach["T"], hard["t_lo"], hard["t_hi"])),
    ]
    explain_figs = [
        draw(
            "hard_importance",
            lambda ax, st: F.importance(ax, st, top_h["label"], top_h["mean_abs_shap"], top_h["direction"]),
        ),
        draw("hard_effects", lambda ax, st: F.effects(ax, st, labels, phi, values)),
    ]
    content = [
        H.kpis(stats),
        f'<p class="muted" style="font-size:12px;margin-top:10px">Blend: P = (1 − a)·S + a·H_S with a = {hard["a_max"]:.2f} × '
        f"ramp(T), rising from 0 at T = {hard['t_lo']} to {hard['a_max']:.2f} at T = {hard['t_hi']}. a_max is the R² of H_S "
        "against the teacher on held-out library molecules, capped at 0.66.</p>",
        H.figures(gate_figs, prefix),
        H.figures(explain_figs, prefix),
        '<h3 style="margin-top:18px; font-size:13px">Top features according to the experiments</h3>'
        + _top_table(top_h, featurizer, bit_dictionary, smarts),
    ]
    return H.section(f"{col['name']} · experiments", "".join(content), hint="hard labels, H_S")
