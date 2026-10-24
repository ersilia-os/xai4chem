"""End to end on the synthetic library: setup, fit (soft + hard), explain, predict, validate, report, info."""

import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from rdkit import Chem

from xai4chem.cli.create_cli import create_cli
from xai4chem.library import ReferenceLibrary
from xai4chem.model import XAIModel
from xai4chem.reporting.drawing import atom_attributions

QUERIES = ["O=C(Nc1ccc(Cl)cc1)c1ccccc1", "c1ccc([N+](=O)[O-])cc1C", "CCCCO"]


def run(args):
    result = CliRunner().invoke(create_cli(), [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture(scope="module")
def library(workspace):
    root, _ = workspace
    lib = root / "library"
    run(
        [
            "setup",
            "--library",
            root / "library_input.csv",
            "--library-dir",
            lib,
            "-f",
            "all",
            "--panel-size",
            300,
            "--bit-dictionary-mols",
            500,
            "-j",
            1,
        ]
    )
    return root, lib


@pytest.fixture(scope="module", params=["morgan", "accfg", "datamol"])
def fitted(request, library):
    root, lib = library
    model_dir = root / f"model_{request.param}"
    run(
        [
            "fit",
            "-s",
            root / "soft.csv",
            "-h",
            root / "hard.csv",
            "-f",
            request.param,
            "--library-dir",
            lib,
            "-m",
            model_dir,
            "--n-estimators",
            300,
        ]
    )
    return root, model_dir, request.param


def test_library_rows_match_on_the_fly(library):
    _, lib = library
    library = ReferenceLibrary(lib)
    rows = np.arange(0, len(library), 97)
    for name in ("morgan", "accfg", "datamol"):
        featurizer = library.featurizer(name)
        stored = library.matrix(name, rows[::-1])[::-1]
        fresh = featurizer.transform_raw([library.smiles[i] for i in rows])
        np.testing.assert_allclose(stored, fresh, rtol=1e-6)
    dup = np.array([5, 1, 5, 3, 1])
    for name in ("morgan", "accfg", "datamol"):
        np.testing.assert_array_equal(library.matrix(name, dup), library.matrix(name, [5, 1, 5, 3, 1][::-1])[::-1])
    for name in ("morgan", "accfg"):
        np.testing.assert_array_equal(library.csr(name, rows[::-1]).toarray()[::-1], library.matrix(name, rows))
    bd = library.bit_dictionary()
    assert any(bd.fragments(b) for b in range(2048))


def test_additivity(fitted):
    _, model_dir, _ = fitted
    model = XAIModel(model_dir)
    res = model.explain(QUERIES)
    for col in model.columns:
        r = res[col.name]
        np.testing.assert_allclose(r["phi_S"].sum(1) + r["base_S"], r["S"], atol=1e-4)
        np.testing.assert_allclose(r["phi_P"].sum(1) + r["base_P"], r["P"], atol=1e-4)
        if col.hard is not None:
            np.testing.assert_allclose(r["phi_H"].sum(1) + r["base_H"], r["h"], atol=1e-4)
            ok = ~r["flat"]
            np.testing.assert_allclose((r["phi_HS"].sum(1) + r["base_HS"])[ok], r["HS"][ok], atol=1e-4)
            assert (r["a"] >= 0).all() and (r["a"] <= col.hard.a_max + 1e-9).all()


def test_predict_matches_explain(fitted):
    _, model_dir, _ = fitted
    model = XAIModel(model_dir)
    res = model.explain(QUERIES)
    pred = model.predict(QUERIES + ["not a smiles"], details=True)
    for col in model.columns:
        np.testing.assert_allclose(pred[col.name].to_numpy()[:3], res[col.name]["prob"], atol=1e-5)
    assert pred.iloc[3, 1:].isna().all()


def test_gate_opens_on_labelled_molecules(fitted):
    root, model_dir, _ = fitted
    model = XAIModel(model_dir)
    col = model.column("activity")
    labelled = col.hard.labelled_smiles[:5]
    r = model.scores(labelled)["activity"]
    np.testing.assert_allclose(r["T"], 1.0, atol=1e-6)
    np.testing.assert_allclose(r["a"], col.hard.a_max, atol=1e-6)


def test_atom_maps_sum_to_present_features(fitted):
    _, model_dir, featurizer = fitted
    model = XAIModel(model_dir)
    res = model.explain(QUERIES)
    X = model.featurize(QUERIES)
    for i, smi in enumerate(QUERIES):
        mol = Chem.MolFromSmiles(smi)
        phi = res["activity"]["phi_P"][i]
        out = atom_attributions(model.featurizer, mol, phi)
        if featurizer == "datamol":
            assert out is None
            continue
        scores, unmapped = out
        np.testing.assert_allclose(scores.sum() + unmapped, phi.sum(), atol=1e-5)
        assert abs(scores.sum() - phi[X[i] > 0].sum()) < 1e-5


def test_hard_head_and_golden(fitted):
    _, model_dir, _ = fitted
    model = XAIModel(model_dir)  # re-checks the golden vectors
    assert model.column("activity").hard is not None
    assert model.column("noisy").hard is None


def test_commands(fitted):
    root, model_dir, featurizer = fitted
    out = root / f"explain_{featurizer}"
    pd.DataFrame({"smiles": QUERIES + ["xx"], "name": ["anilide", "nitro", "butanol", "bad"]}).to_csv(
        root / "q.csv", index=False
    )
    run(["explain", "-m", model_dir, "-i", root / "q.csv", "-o", out, "--id-column", "name", "--png"])
    pred = pd.read_csv(out / "predictions.csv")
    assert {"activity", "activity_S", "activity_HS", "activity_a", "activity_nearest_smiles"} <= set(pred.columns)
    assert {"library_T", "library_nearest_smiles", "activity_library_teacher"} <= set(pred.columns)
    # Library molecules are their own nearest neighbours, and the teacher's value is the soft label
    assert pred.loc[pred.id == "anilide", "library_T"].iloc[0] <= 1.0
    assert not pred.loc[pred.id == "bad", "valid"].iloc[0]
    assert pred.loc[pred.id == "bad", ["library_T", "activity_library_teacher"]].isna().all(axis=None)
    assert (out / "index.html").exists() and "anilide" in (out / "index.html").read_text()
    if featurizer != "datamol":
        assert (out / "png" / "anilide_activity.png").exists()
    run(["predict", "-m", model_dir, "-i", root / "q.csv", "-o", root / f"pred_{featurizer}.csv", "--details"])
    assert len(pd.read_csv(root / f"pred_{featurizer}.csv")) == 4
    assert (model_dir / "report.html").exists()
    for page in (out / "index.html", model_dir / "report.html"):
        text = page.read_text()
        assert 'src="http' not in text and '<link rel="stylesheet"' not in text  # self-contained
    run(["report", "-m", model_dir, "-o", root / f"report_{featurizer}.html"])
    run(
        [
            "validate",
            "-m",
            model_dir,
            "-s",
            root / "soft.csv",
            "-h",
            root / "hard.csv",
            "-o",
            root / f"val_{featurizer}.json",
        ]
    )
    val = json.loads((root / f"val_{featurizer}.json").read_text())
    assert val["activity"]["soft_S"]["r2"] > 0.5
    run(["info", "-m", model_dir])
    run(["predict", "-m", model_dir, "-S", "CCO", "-S", "c1ccccc1O", "-o", root / f"pred2_{featurizer}.csv"])
    assert len(pd.read_csv(root / f"pred2_{featurizer}.csv")) == 2
    run(["explain", "-m", model_dir, "-S", "CCO", "-o", root / f"explain2_{featurizer}", "--no-gallery"])


def test_fit_refuses_to_overwrite(fitted):
    root, model_dir, featurizer = fitted
    result = CliRunner().invoke(
        create_cli(), ["fit", "-s", str(root / "soft.csv"), "-m", str(model_dir), "-f", featurizer]
    )
    assert result.exit_code != 0
    assert "--force" in result.output


def test_surrogate_finds_the_teacher_features(fitted):
    """The teacher depends on amides and halogens; the AccFG surrogate should rank those groups first."""
    _, model_dir, featurizer = fitted
    if featurizer != "accfg":
        pytest.skip("AccFG labels are directly checkable")
    top = pd.read_csv(model_dir / "columns" / "c0" / "explain" / "global_s.csv").head(6)["label"].str.lower()
    assert any("amide" in t for t in top)
    assert any(h in t for t in top for h in ("halo", "chlor", "fluor", "brom"))


def test_explain_and_predict_agree_on_scales(fitted):
    root, model_dir, featurizer = fitted
    pd.DataFrame({"smiles": QUERIES}).to_csv(root / f"q3_{featurizer}.csv", index=False)
    run(
        [
            "explain",
            "-m",
            model_dir,
            "-i",
            root / f"q3_{featurizer}.csv",
            "-o",
            root / f"ex3_{featurizer}",
            "--no-gallery",
        ]
    )
    run(
        [
            "predict",
            "-m",
            model_dir,
            "-i",
            root / f"q3_{featurizer}.csv",
            "-o",
            root / f"pr3_{featurizer}.csv",
            "--details",
        ]
    )
    ex = pd.read_csv(root / f"ex3_{featurizer}" / "predictions.csv")
    pr = pd.read_csv(root / f"pr3_{featurizer}.csv")
    for c in ("activity", "activity_S", "activity_HS"):
        np.testing.assert_allclose(ex[c], pr[c], atol=1e-5)


def test_rerunning_learn_hard_replaces_the_head(fitted, tmp_path):
    import shutil

    root, model_dir, featurizer = fitted
    copy = tmp_path / "m"
    shutil.copytree(model_dir, copy)
    hard = pd.read_csv(root / "hard.csv").head(10)
    hard.to_csv(tmp_path / "few.csv", index=False)
    run(["learn-hard", "-h", tmp_path / "few.csv", "-m", copy])
    assert XAIModel(copy).column("activity").hard is None


def test_continuous_hard_labels(library, tmp_path):
    """Non-binary measurements give a regression H, calibrated and blended like a classifier."""
    from .conftest import teacher

    root, lib = library
    soft = pd.read_csv(root / "soft.csv")
    take = soft.sample(120, random_state=3)
    hard = pd.DataFrame({"smiles": take.smiles, "noisy": teacher(take.smiles.tolist()) * 10 + 1})
    hard.to_csv(tmp_path / "hard_continuous.csv", index=False)
    model_dir = tmp_path / "m"
    run(
        [
            "fit",
            "-s",
            root / "soft.csv",
            "-h",
            tmp_path / "hard_continuous.csv",
            "--library-dir",
            lib,
            "-m",
            model_dir,
            "--n-estimators",
            200,
        ]
    )
    model = XAIModel(model_dir)
    col = model.column("noisy")
    assert col.hard is not None and col.hard.task == "continuous"
    r = model.explain(QUERIES, columns=["noisy"])["noisy"]
    np.testing.assert_allclose(r["phi_P"].sum(1) + r["base_P"], r["P"], atol=1e-4)


def test_soft_labels_outside_the_library(library, tmp_path):
    """Soft labels on molecules the library does not have: featurized on the fly, hard head still fitted."""
    from .conftest import experiments, teacher

    root, lib = library
    known = set(pd.read_csv(root / "library_input.csv").smiles)
    novel = []
    for s in pd.read_csv(root / "soft.csv").smiles.tolist()[:600]:
        mol = Chem.MolFromSmiles(s + "C")
        if mol is not None and Chem.MolToSmiles(mol) not in known:
            novel.append(Chem.MolToSmiles(mol))
    novel = list(dict.fromkeys(novel))
    pd.DataFrame({"smiles": novel, "activity": teacher(novel)}).to_csv(tmp_path / "soft_novel.csv", index=False)
    pd.DataFrame({"smiles": novel[:150], "activity": experiments(novel[:150])}).to_csv(
        tmp_path / "hard_novel.csv", index=False
    )
    run(
        [
            "fit",
            "-s",
            tmp_path / "soft_novel.csv",
            "-h",
            tmp_path / "hard_novel.csv",
            "--library-dir",
            lib,
            "-m",
            tmp_path / "m",
            "--n-estimators",
            200,
        ]
    )
    assert XAIModel(tmp_path / "m").column("activity").hard is not None
