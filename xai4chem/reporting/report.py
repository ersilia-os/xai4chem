"""A self-contained HTML report of a model: fidelity, what the teacher has learned, and the hard head."""

import time
from pathlib import Path

import pandas as pd

from xai4chem.console import STEP_COLORS, elapsed, rule, set_active_color, summary_panel
from xai4chem.console import path as cpath
from xai4chem.reporting import html as H
from xai4chem.reporting.drawing import fragment_svg
from xai4chem.run import EXPLAIN_DIRNAME, column_dir, read_json, read_manifest

TOP = 20

METHODS = """
<details style="margin-top:8px"><summary>How to read this report</summary>
<div class="card" style="margin-top:10px; font-size:12.5px; line-height:1.55">
<p><b>S</b> is a gradient-boosting surrogate trained to reproduce the teacher's outputs (soft labels) over the
reference library. Probability outputs are learned in logit space. Its <b>fidelity</b> is measured on library
molecules held out from training. Explanations describe S, so they are as trustworthy as that fidelity.</p>
<p>Attributions are exact TreeSHAP values on S's scale. <b>Importance</b> is the mean |SHAP| over the explanation
panel, a fixed random subset of the library. <b>Direction</b> is the correlation between a feature's value and its
SHAP value: positive means more of the feature raises the prediction. <b>When present</b> is the mean |SHAP| over the
molecules that contain the feature, which ranks rare but strong features fairly.</p>
__MORGAN__<p>With hard labels, <b>H</b> is trained on the measurements with the same features, and carried onto S's scale by
an isotonic map (<b>H_S</b>). The prediction is P = (1 − a)·S + a·H_S, where a = a_max · ramp(T) and T is the
Tanimoto similarity (binary Morgan) to the nearest labelled molecule: the hard head only speaks near the labelled
chemistry. a_max is earned from how well H_S reproduces the teacher on held-out library molecules. Because a is
fixed for a given molecule, the explanation of P is exactly (1 − a)·φ_S + a·φ_HS.</p>
</div></details>
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


def _top_table(df, featurizer_name, bit_dictionary=None, smarts=None):
    rows = []
    counts = featurizer_name != "datamol"
    vmax = float(df["mean_abs_shap"].head(TOP).max()) or 1.0
    for _, r in df.head(TOP).iterrows():
        drawing = ""
        if featurizer_name == "morgan" and bit_dictionary is not None:
            bit = int(str(r["feature"]).split("-")[1])
            frags = bit_dictionary.get(str(bit), {}).get("fragments", [])
            if frags:
                drawing = fragment_svg(frags[0]["smiles"], size=64)
        elif featurizer_name == "accfg" and smarts is not None:
            idx = int(str(r["feature"]).split("-")[1])
            drawing = fragment_svg(smarts[idx], size=64, smarts=True)
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
    # Presence only means something for count features; descriptors are (almost) never zero
    headers = ["", "Feature", "Importance", "Direction"] + (["Frequency", "When present"] if counts else [])
    if "collision" in df.columns:
        headers.append("Collision")
    return H.table(headers, rows, numeric=(2, 4, 5, 6) if counts else (2,))


def _chemotype_html(chemo):
    rows = []
    for _, r in chemo.iterrows():
        drawing = fragment_svg(r["scaffold"], size=64)
        rows.append(
            [
                f'<div class="frag">{drawing}</div>' if drawing else "",
                f'<span class="smi">{H.esc(r["scaffold"])}</span>',
                f"{int(r['n']):,}",
                _f(r["mean_teacher"]),
                _f(r["mean_S"]),
                H.esc(r["top_features"]),
            ]
        )
    return H.table(["", "Scaffold", "Molecules", "Teacher", "S", "Distinctive drivers"], rows, numeric=(2, 3, 4))


def _training_details(model_dir, col):
    """A collapsed block with what S was trained on and how: rows, trees, scale, boosting parameters."""
    meta = read_json(column_dir(model_dir, col["id"]) / "soft_meta.json")
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


def _substructure_rows(table, bit_dictionary):
    """Rows of a global table whose bit is named by an environment of radius ≥ 1."""
    if bit_dictionary is None:
        return None

    def radius(feature):
        frags = bit_dictionary.get(str(feature).split("-")[1], {}).get("fragments", [])
        return frags[0]["radius"] if frags else 0

    return table[[radius(f) >= 1 for f in table["feature"]]].reset_index(drop=True)


MORGAN_NOTE = """<p>Morgan bits are folded, so one bit can be set by several substructures. Each bit is named by the
substructure that most often sets it in the library, with that substructure's share; <b>collision</b> is the share of
occurrences that come from other substructures.</p>
"""


def write_report(model_dir, output, open_browser=False):
    started = time.time()
    color = STEP_COLORS["report"]
    set_active_color(color)
    model_dir = Path(model_dir)
    output = Path(output)
    if output.suffix != ".html":
        output = output / "report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(model_dir)
    rule("xai4chem · report", style=color, right=cpath(model_dir))
    featurizer = manifest["featurizer"]["name"]
    bit_dictionary = None
    if (model_dir / "bit_dictionary.json").exists():
        bit_dictionary = read_json(model_dir / "bit_dictionary.json")["bits"]
    smarts = None
    if featurizer == "accfg":
        from xai4chem.featurizers.accfg import AccFgFingerprint

        smarts = AccFgFingerprint().smarts()

    body = []
    for col in manifest["columns"]:
        exp = column_dir(model_dir, col["id"]) / EXPLAIN_DIRNAME
        if not (exp / "summary.json").exists():
            raise FileNotFoundError(
                f"No global explanations for {col['name']!r}. Run `xai4chem explain-global -m {model_dir}` first."
            )
        s = read_json(exp / "summary.json")
        fid = s["fidelity"]
        hard = s.get("hard")
        stats = [
            H.stat(
                "Fidelity R²",
                _f(fid["r2"]),
                f"{fid['n']:,} held-out library molecules",
                tip="R² of S against the teacher, on S's scale",
            ),
            H.stat("Spearman ρ", _f(fid["spearman"]), "rank agreement with the teacher"),
        ]
        if s.get("stability_spearman") is not None:
            stats.append(H.stat("Stability", _f(s["stability_spearman"], "{:.2f}"), "importances across two seeds"))
        if hard:
            cv = hard["cv"]
            cv_v, cv_k = (cv["auroc"], "CV AUROC") if "auroc" in cv else (cv["spearman"], "CV Spearman")
            stats.append(H.stat(cv_k, _f(cv_v), f"H on {hard['n_labels']:,} labels"))
            stats.append(
                H.stat(
                    "a_max", _f(hard["a_max"], "{:.2f}"), f"R² of H_S vs teacher {_f(hard['alignment_r2'], '{:.2f}')}"
                )
            )
        content = [H.kpis(stats)]
        figs = []
        if (exp / "s_fidelity.png").exists():
            figs.append(H.figure(exp / "s_fidelity.png", "Teacher against S on held-out library molecules."))
        figs.append(
            H.figure(
                exp / "s_beeswarm.png",
                "Effect of the top features; each dot is one panel molecule, coloured by the feature's value.",
            )
        )
        content.append(f'<div class="figrow2" style="margin-top:14px">{"".join(figs)}</div>')
        top_s = pd.read_csv(exp / "global_s.csv")
        content.append(
            '<h3 style="margin-top:18px; font-size:13px">Top features</h3>'
            + _top_table(top_s, featurizer, bit_dictionary, smarts)
        )
        if (exp / "chemotypes.csv").exists():
            chemo = pd.read_csv(exp / "chemotypes.csv")
            if len(chemo):
                content.append(
                    '<h3 style="margin-top:18px; font-size:13px">Chemotypes</h3>'
                    '<p class="faint" style="font-size:11.5px;margin:0 0 6px">The most common Bemis–Murcko scaffolds '
                    "of the panel: how the teacher and S score them, and the features that matter more within each scaffold "
                    "than across the panel.</p>" + _chemotype_html(chemo)
                )
        substructures = _substructure_rows(top_s, bit_dictionary) if featurizer == "morgan" else None
        if substructures is not None and len(substructures):
            content.append(
                '<h3 style="margin-top:18px; font-size:13px">Top substructures</h3>'
                '<p class="faint" style="font-size:11.5px;margin:0 0 6px">Bits whose named environment spans at '
                "least one bond. Single-atom bits (above) mostly track molecular size and composition.</p>"
                + _top_table(substructures, featurizer, bit_dictionary, smarts)
            )
        content.append(_training_details(model_dir, col))
        body.append(
            H.section(
                f"{col['name']} · what the teacher has learned",
                "".join(content),
                hint=f"panel of {s['panel_size']:,} library molecules",
            )
        )

        if hard:
            hc = [
                H.kpis(
                    [
                        H.stat("Task", hard["task"], f"{hard['n_labels']:,} labelled molecules"),
                        H.stat(
                            "Reach",
                            _f(hard.get("frac_panel_gated"), "{:.1%}"),
                            f"of the library has a > 0 (T ≥ {hard['t_lo']})",
                        ),
                        H.stat(
                            "Agreement",
                            _f(hard.get("importance_spearman_s_vs_hs"), "{:.2f}"),
                            "S vs H_S importances (ρ)",
                        ),
                        H.stat("Calibration", hard["isotonic_direction"], "isotonic H → S map"),
                    ]
                ),
                (
                    f'<p class="muted" style="font-size:12px;margin-top:10px">Blend: P = (1 − a)·S + a·H_S with '
                    f"a = {hard['a_max']:.2f} × ramp(T), rising from 0 at T = {hard['t_lo']} to {hard['a_max']:.2f} "
                    f"at T = {hard['t_hi']} (Tanimoto to the nearest labelled molecule). a_max is the R² of H_S "
                    "against the teacher on held-out library molecules, capped at 0.66.</p>"
                ),
            ]
            hfigs = [
                H.figure(
                    exp / "hs_calibration.png",
                    "Isotonic map from H's margin onto S's scale; ticks are labelled molecules.",
                ),
                H.figure(
                    exp / "hs_reach.png",
                    "Similarity of library molecules to the labelled set; dashed lines are the ramp of the gate.",
                ),
            ]
            hc.append(f'<div class="figrow2" style="margin-top:14px">{"".join(hfigs)}</div>')
            top_h = pd.read_csv(exp / "global_hs.csv")
            hc.append(
                '<h3 style="margin-top:18px; font-size:13px">Top features according to the experiments</h3>'
                + _top_table(top_h, featurizer, bit_dictionary, smarts)
            )
            body.append(H.section(f"{col['name']} · experiments", "".join(hc), hint="hard labels, H_S"))

    lede = (
        f"Surrogate of a teacher model, learned from soft labels over the {H.esc(manifest['library']['name'])} reference library "
        f"({manifest['library']['n_molecules']:,} molecules) with {H.esc(manifest['featurizer']['name'])} features. "
        f"xai4chem {H.esc(manifest['xai4chem_version'])}."
    )
    page = H.page(
        f"xai4chem · {model_dir.name}",
        "xai4chem",
        f"{H.esc(model_dir.name)} · model <em>report</em>",
        lede,
        METHODS.replace("__MORGAN__", MORGAN_NOTE if featurizer == "morgan" else "") + "".join(body),
    )
    output.write_text(page)
    summary_panel(
        "xai4chem · report",
        [
            ("Report", f"[dim]{cpath(output)}[/]"),
            ("Size", f"{output.stat().st_size / 1e6:.1f} MB"),
            ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
        ],
        border_style=color,
        icon="✓",
    )
    if open_browser:
        import webbrowser

        webbrowser.open(output.resolve().as_uri())
    return output
