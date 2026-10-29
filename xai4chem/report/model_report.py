"""``report.html``: fidelity, what the teacher has learned, and the hard head, as one self-contained page."""

from pathlib import Path

import numpy as np
import pandas as pd

from xai4chem.report import html as H
from xai4chem.report import svg
from xai4chem.report.drawing import fragment_svg
from xai4chem.run import (
    EXPLAIN_DIRNAME,
    FIDELITY_NAME,
    H_TO_S_NAME,
    HARD_DIRNAME,
    REPORT_NAME,
    column_dir,
    read_json,
    read_manifest,
)

TOP = 20

METHODS = """
<details style="margin-top:8px"><summary>How to read this report</summary>
<div class="card" style="margin-top:10px; font-size:12.5px; line-height:1.55">
<p><b>S</b> is a gradient-boosting surrogate trained to reproduce the teacher's outputs (soft labels) over the
reference library; probability outputs are learned in logit space. Its <b>fidelity</b> is measured on library
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


def _effects(plot_path):
    data = np.load(plot_path)
    return svg.effects(data["phi"], data["values"], list(data["labels"]))


def _training_details(meta):
    rows = [
        ["Scale", H.esc(meta["transform"])],
        ["Training rows", f"{meta['n_train']:,}"],
        ["Held-out rows", f"{meta['n_val']:,}"],
        ["Trees", f"{meta['n_trees']:,}"],
        [
            "Fidelity on the teacher's scale",
            f"R² {_f(meta['fidelity_teacher_scale']['r2'])} · RMSE {_f(meta['fidelity_teacher_scale']['rmse'])}",
        ],
    ]
    rows += [[H.esc(k), H.esc(v)] for k, v in meta["params"].items()]
    return (
        '<details style="margin-top:14px"><summary>Training details</summary>' + H.table(["", ""], rows) + "</details>"
    )


def write_report(model_dir, output=None):
    """Write the model report from what ``fit`` saved in the model directory.

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
    body = []
    for col in manifest["columns"]:
        exp = column_dir(model_dir, col["id"]) / EXPLAIN_DIRNAME
        hard = read_json(exp / "summary.json").get("hard")
        body.append(_soft_section(model_dir, col, featurizer, bit_dictionary, smarts))
        if hard:
            body.append(_hard_section(col, hard, exp, featurizer, bit_dictionary, smarts))
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


def _soft_section(model_dir, col, featurizer, bit_dictionary, smarts):
    """What the teacher has learned, for one column: fidelity, figures, top features, training details."""
    cdir = column_dir(model_dir, col["id"])
    exp = cdir / EXPLAIN_DIRNAME
    s = read_json(exp / "summary.json")
    meta = read_json(cdir / "soft_meta.json")
    fid, hard = s["fidelity"], s.get("hard")
    scale = " · logit scale" if col["transform"] == "logit" else ""
    stats = [
        H.stat("Fidelity R²", _f(fid["r2"]), f"{fid['n']:,} held-out library molecules{scale}"),
        H.stat("Spearman ρ", _f(fid["spearman"]), "rank agreement with the teacher"),
    ]
    if hard:
        ho = hard["heldout"]
        key, value = ("Held-out AUROC", ho["auroc"]) if "auroc" in ho else ("Held-out ρ", ho["spearman"])
        stats.append(H.stat(key, _f(value), f"H on {hard['n_labels']:,} labels"))
        stats.append(
            H.stat("a_max", _f(hard["a_max"], "{:.2f}"), f"R² of H_S vs teacher {_f(hard['alignment_r2'], '{:.2f}')}")
        )
    top_s = pd.read_csv(exp / "global_s.csv")
    figs = []
    if (cdir / FIDELITY_NAME).exists():
        data = np.load(cdir / FIDELITY_NAME)
        figs.append(
            H.figure(
                svg.density(data["teacher"], data["surrogate"], f"Teacher{scale}", f"Surrogate S{scale}"),
                "Teacher against S on held-out library molecules.",
            )
        )
    figs.append(H.figure(_effects(exp / "plot_s.npz"), "Effect of the top features on panel molecules."))
    content = [H.kpis(stats), f'<div class="figrow2" style="margin-top:14px">{"".join(figs)}</div>']
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


def _hard_section(col, hard, exp, featurizer, bit_dictionary, smarts):
    reach = np.load(exp / "reach.npz")
    stats = [
        H.stat("Task", hard["task"], f"{hard['n_labels']:,} labelled molecules"),
        H.stat("Reach", _f(hard["reach"], "{:.1%}"), f"of panel molecules have a > 0 (T ≥ {hard['t_lo']})"),
        H.stat("Agreement", _f(hard["importance_spearman_s_vs_hs"], "{:.2f}"), "S vs H_S importances (ρ)"),
        H.stat("Calibration", hard["isotonic_direction"], "isotonic H → S map"),
    ]
    iso = read_json(Path(exp).parent / HARD_DIRNAME / H_TO_S_NAME)
    figs = [
        H.figure(
            svg.curve(iso["x"], iso["y"], reach["h"], "H margin", "H_S (S scale)"),
            "Isotonic map from H's margin onto S's scale; ticks are labelled molecules.",
        ),
    ]
    if len(reach["T"]):
        figs.append(
            H.figure(
                svg.histogram(
                    reach["T"],
                    "1-NN Tanimoto to the labelled set",
                    "Panel molecules",
                    vlines=(hard["t_lo"], hard["t_hi"]),
                ),
                "Similarity of panel molecules to the labelled set; dashed lines bound the gate's ramp.",
            )
        )
    top_h = pd.read_csv(exp / "global_hs.csv")
    content = [
        H.kpis(stats),
        f'<p class="muted" style="font-size:12px;margin-top:10px">Blend: P = (1 − a)·S + a·H_S with a = {hard["a_max"]:.2f} × '
        f"ramp(T), rising from 0 at T = {hard['t_lo']} to {hard['a_max']:.2f} at T = {hard['t_hi']}. a_max is the R² of H_S "
        "against the teacher on held-out library molecules, capped at 0.66.</p>",
        f'<div class="figrow2" style="margin-top:14px">{"".join(figs)}</div>',
        H.figure(
            _effects(exp / "plot_hs.npz"),
            "What the experiments say: effect of the top features on the labelled molecules.",
        ),
        '<h3 style="margin-top:18px; font-size:13px">Top features according to the experiments</h3>'
        + _top_table(top_h, featurizer, bit_dictionary, smarts),
    ]
    return H.section(f"{col['name']} · experiments", "".join(content), hint="hard labels, H_S")
