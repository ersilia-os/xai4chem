"""End to end on the synthetic library: setup, fit (soft + hard), explain and predict, through the CLI and the API."""

import json
import re
import shutil

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from rdkit import Chem

from xai4chem import api
from xai4chem.cli.create_cli import create_cli
from xai4chem.data.library import ReferenceLibrary, library_for_model
from xai4chem.explain.atoms import atom_attributions
from xai4chem.models.model import XAIModel

QUERIES = ["O=C(Nc1ccc(Cl)cc1)c1ccccc1", "c1ccc([N+](=O)[O-])cc1C", "CCCCO"]
TREES = 300


def run(args):
    result = CliRunner().invoke(create_cli(), [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def fails(args, *needles):
    result = CliRunner().invoke(create_cli(), [str(a) for a in args])
    assert result.exit_code != 0, result.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)  # CI forces colour, which splits phrases
    text = " ".join(plain.replace("│", " ").split())  # the error panel wraps and boxes lines
    for needle in needles:
        assert needle in text, text
    return result


@pytest.fixture(scope="module")
def library(workspace):
    root, _ = workspace
    lib = root / "library"
    result = run(["setup", "--library", root / "library_input.csv", "--library-dir", lib, "-f", "all"])
    assert result.output.count("xai4chem · setup") == 2  # one rule, one panel
    return root, lib


@pytest.fixture(scope="module", params=["morgan", "accfg", "datamol"])
def fitted(request, library):
    root, lib = library
    model_dir = root / f"model_{request.param}"
    api.fit(root / "soft.csv", model_dir, root / "hard.csv", request.param, library_dir=lib, n_estimators=TREES)
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
    assert not pred["valid"].iloc[3] and pred.iloc[3, 2:].isna().all()


def test_gate_opens_on_labelled_molecules(fitted):
    _, model_dir, _ = fitted
    model = XAIModel(model_dir)
    col = model.column("activity")
    r = model.scores(col.hard.labelled_smiles[:5])["activity"]
    np.testing.assert_allclose(r["T"], 1.0, atol=1e-6)
    np.testing.assert_allclose(r["a"], col.hard.a_max, atol=1e-6)


def test_atom_maps_sum_to_present_features(fitted):
    _, model_dir, featurizer = fitted
    model = XAIModel(model_dir)
    res = model.explain(QUERIES)
    X = model.featurize(QUERIES)
    for i, smi in enumerate(QUERIES):
        phi = res["activity"]["phi_P"][i]
        out = atom_attributions(model.featurizer, Chem.MolFromSmiles(smi), phi)
        if featurizer == "datamol":
            assert out is None
            continue
        scores, unmapped = out
        np.testing.assert_allclose(scores.sum() + unmapped, phi.sum(), atol=1e-5)
        assert abs(scores.sum() - phi[X[i] > 0].sum()) < 1e-5


def test_model_directory(fitted):
    _, model_dir, _ = fitted
    model = XAIModel(model_dir)  # re-checks the golden vectors
    assert model.column("activity").hard is not None
    assert model.column("noisy").hard is None
    summary = json.loads((model_dir / "columns" / "c0" / "explain" / "summary.json").read_text())
    assert 0 <= summary["hard"]["reach"] <= 1 and summary["fidelity"]["r2"] > 0.5
    text = (model_dir / "report.html").read_text()
    assert "<svg" in text and 'src="http' not in text and '<link rel="stylesheet"' not in text  # self-contained


def test_explain_and_predict_commands(fitted):
    root, model_dir, featurizer = fitted
    out = root / f"explain_{featurizer}"
    q = root / f"q_{featurizer}.csv"
    pd.DataFrame({"smiles": QUERIES + ["xx"], "name": ["anilide", "nitro", "butanol", "bad"]}).to_csv(q, index=False)
    result = run(["explain", "-m", model_dir, "-i", q, "-o", out, "--domain"])
    assert result.output.count("xai4chem · explain") == 2  # one rule, one panel
    pred = pd.read_csv(out / "predictions.csv")
    assert {"activity", "activity_S", "activity_HS", "activity_a", "activity_nearest_smiles"} <= set(pred.columns)
    assert {"library_T", "library_nearest_smiles", "activity_library_teacher"} <= set(pred.columns)
    assert pred.id.tolist() == ["anilide", "nitro", "butanol", "bad"]  # ids from the `name` column
    assert not pred.loc[pred.id == "bad", "valid"].iloc[0]
    assert pred.loc[pred.id == "bad", ["library_T", "activity_library_teacher"]].isna().all(axis=None)
    page = (out / "index.html").read_text()
    assert "anilide" in page and 'src="http' not in page
    run(["predict", "-m", model_dir, "-i", q, "-o", root / f"pred_{featurizer}.csv"])
    pr = pd.read_csv(root / f"pred_{featurizer}.csv")
    assert len(pr) == 4
    np.testing.assert_allclose(pr["activity"][:3], pred["activity"][:3], atol=1e-5)
    run(["predict", "-m", model_dir, "-S", "CCO", "-S", "c1ccccc1O", "-o", root / f"pred2_{featurizer}.csv"])
    assert len(pd.read_csv(root / f"pred2_{featurizer}.csv")) == 2


def test_two_molecules_are_compared_exactly(fitted, tmp_path):
    _, model_dir, _ = fitted
    result = run(["explain", "-m", model_dir, "-S", QUERIES[0], "-S", QUERIES[1], "-o", tmp_path / "e"])
    assert "gap" in result.output
    assert "vs mol_0001" in (tmp_path / "e" / "index.html").read_text()
    out = api.explain(model_dir, QUERIES[:2], tmp_path / "e2")
    for pair in out["comparisons"]:
        assert abs(pair["residual"]) < 1e-4


def test_surrogate_finds_the_teacher_features(fitted):
    """The teacher depends on amides and halogens; the AccFG surrogate should rank those groups first."""
    _, model_dir, featurizer = fitted
    if featurizer != "accfg":
        pytest.skip("AccFG labels are directly checkable")
    top = pd.read_csv(model_dir / "columns" / "c0" / "explain" / "global_s.csv").head(6)["label"].str.lower()
    assert any("amide" in t for t in top)
    assert any(h in t for t in top for h in ("halo", "chlor", "fluor", "brom"))


def test_fit_command(library, tmp_path):
    root, lib = library
    args = ["fit", "-s", root / "soft.csv", "-h", root / "hard.csv", "--library-dir", lib, "-m", tmp_path / "m"]
    result = run(args)
    assert result.output.count("xai4chem · fit") == 2  # one rule, one panel
    assert "Step 5/5" in result.output and (tmp_path / "m" / "report.html").exists()
    fails(args, "--force")
    run(args + ["--force"])


def test_fit_and_predict_options_reach_the_model(library, tmp_path):
    root, lib = library
    m = tmp_path / "m"
    flags = ["--n-estimators", "80", "--panel-rows", "50", "--seed", "7", "--reweight", "off"]
    run(["fit", "-s", root / "soft.csv", "--library-dir", lib, "-m", m, "--no-report", *flags])
    assert not (m / "report.html").exists()
    meta = json.loads((m / "columns" / "c0" / "soft_meta.json").read_text())
    assert meta["seed"] == 7 and meta["reweight"]["strategy"] == "none"
    assert meta["n_trees"] <= 80
    run(["predict", "-m", m, "-S", "CCO", "--details", "-o", tmp_path / "p.csv"])
    assert "activity_S" in pd.read_csv(tmp_path / "p.csv").columns


def test_regression_hard_labels(library, tmp_path):
    """Non-binary measurements give a regression H, calibrated and blended like a classifier."""
    from .conftest import teacher

    root, lib = library
    take = pd.read_csv(root / "soft.csv").sample(120, random_state=3)
    hard = pd.DataFrame({"smiles": take.smiles, "noisy": teacher(take.smiles.tolist()) * 10 + 1})
    hard.to_csv(tmp_path / "hard_regression.csv", index=False)
    api.fit(root / "soft.csv", tmp_path / "m", tmp_path / "hard_regression.csv", library_dir=lib, n_estimators=200)
    model = XAIModel(tmp_path / "m")
    col = model.column("noisy")
    assert col.hard is not None and col.hard.task == "regression"
    r = model.explain(QUERIES, columns=["noisy"])["noisy"]
    np.testing.assert_allclose(r["phi_P"].sum(1) + r["base_P"], r["P"], atol=1e-4)


def test_forced_binary_task_is_refused_before_training(library, tmp_path):
    root, lib = library
    take = pd.read_csv(root / "soft.csv").sample(120, random_state=3)
    pd.DataFrame({"smiles": take.smiles, "noisy": take.noisy}).to_csv(tmp_path / "hard_scores.csv", index=False)
    args = ["fit", "-s", root / "soft.csv", "-h", tmp_path / "hard_scores.csv", "--library-dir", lib]
    fails(args + ["-m", tmp_path / "m", "--task", "binary"], "forced to --task binary")
    assert not (tmp_path / "m" / "manifest.json").exists()


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
    pd.DataFrame({"smiles": novel, "activity": teacher(novel)}).to_csv(tmp_path / "soft.csv", index=False)
    pd.DataFrame({"smiles": novel[:150], "activity": experiments(novel[:150])}).to_csv(
        tmp_path / "hard.csv", index=False
    )
    api.fit(tmp_path / "soft.csv", tmp_path / "m", tmp_path / "hard.csv", library_dir=lib, n_estimators=200)
    assert XAIModel(tmp_path / "m").column("activity").hard is not None


def test_unlearnable_teacher_columns_are_refused_before_training(library, tmp_path):
    """Olinda refuses an empty or constant column instead of skipping it, so no model is half-written."""
    root, lib = library
    soft = pd.read_csv(root / "soft.csv")
    for name, values, needle in (("broken", np.nan, "0 finite value(s)"), ("flat", 0.5, "is constant")):
        path = tmp_path / f"soft_{name}.csv"
        soft.assign(**{name: values}).to_csv(path, index=False)
        fails(["fit", "-s", path, "--library-dir", lib, "-m", tmp_path / name], needle)
        assert not (tmp_path / name).exists()


def test_friendly_refusals(library, tmp_path):
    root, lib = library
    soft = pd.read_csv(root / "soft.csv")
    soft.assign(smiles=[f"CMP{i}" for i in range(len(soft))]).rename(columns={"smiles": "compound_id"}).to_csv(
        tmp_path / "no_smiles.csv", index=False
    )
    fails(
        ["fit", "-s", tmp_path / "no_smiles.csv", "--library-dir", lib, "-m", tmp_path / "a"],
        "does not look like SMILES",
    )
    pd.read_csv(root / "hard.csv").rename(columns={"activity": "Activity"}).to_csv(
        tmp_path / "hard_bad.csv", index=False
    )
    fails(
        ["fit", "-s", root / "soft.csv", "-h", tmp_path / "hard_bad.csv", "--library-dir", lib, "-m", tmp_path / "b"],
        "matches a soft-label column",
    )
    assert not (tmp_path / "b" / "manifest.json").exists()  # refused before any training
    fails(["setup", "--library", tmp_path / "no_smiles.csv", "--library-dir", lib], "already holds the library")
    fails(["explain", "-m", tmp_path / "nope", "-S", "CCO"], "Train one with")
    fails(["predict", "-m", tmp_path / "nope", "-S", "CCO", "-i", root / "soft.csv"], "not both")
    fails(["fit", "-s", root / "soft.csv", "--library-dir", tmp_path / "none", "-m", tmp_path / "c"], "xai4chem setup")


def test_all_invalid_smiles_is_refused(fitted, tmp_path):
    _, model_dir, _ = fitted
    fails(["explain", "-m", model_dir, "-S", "C1CC(", "-o", tmp_path / "e"], "nothing to explain")


def test_library_is_relocatable(library, fitted, tmp_path):
    """A library directory can be copied elsewhere (e.g. through eosvc) and models still find it."""
    _, lib = library
    _, model_dir, featurizer = fitted
    moved = tmp_path / "moved_library"
    shutil.copytree(lib, moved)
    copy = ReferenceLibrary(moved)
    np.testing.assert_array_equal(copy.matrix(featurizer, [3, 1]), ReferenceLibrary(lib).matrix(featurizer, [3, 1]))
    assert library_for_model(XAIModel(model_dir).manifest, moved).dir == moved


def test_library_maintenance(library, fitted, tmp_path):
    root, lib = library
    _, model_dir, _ = fitted
    moved = tmp_path / "moved"
    shutil.copytree(lib, moved)
    run(["setup", "--library-dir", moved, "--export-smiles", tmp_path / "smiles.csv"])  # extending needs no --library
    assert pd.read_csv(tmp_path / "smiles.csv").smiles.tolist() == ReferenceLibrary(lib).smiles
    # A library from an older xai4chem is refused with the way out
    manifest = json.loads((moved / "library.json").read_text())
    (moved / "library.json").write_text(json.dumps({**manifest, "schema": "xai4chem.library.v1"}))
    with pytest.raises(ValueError, match="setup --force"):
        ReferenceLibrary(moved)
    # An explicit library that does not match the model is refused, not silently replaced
    pd.DataFrame({"smiles": QUERIES}).to_csv(tmp_path / "tiny.csv", index=False)
    api.setup(tmp_path / "tiny.csv", library_dir=tmp_path / "tiny_lib")
    with pytest.raises(ValueError, match="was trained on"):
        library_for_model(XAIModel(model_dir).manifest, tmp_path / "tiny_lib")
    result = run(["predict", "-m", model_dir, "-S", "CCO", "-S", "notasmiles", "-o", tmp_path / "p.csv"])
    assert "1 unparseable" in result.output


def test_published_library_is_downloaded(library, tmp_path, monkeypatch):
    """`setup` with no --library downloads the published files, checks them and builds the cache locally."""
    from xai4chem.data.build import publish_manifest

    _, lib = library
    remote = tmp_path / "bucket"
    shutil.copytree(lib, remote, ignore=shutil.ignore_patterns("morgan_bits_*"))
    publish_manifest(remote)
    monkeypatch.setenv("XAI4CHEM_LIBRARY_URL", str(remote))
    local = tmp_path / "local"
    result = run(["setup", "--library-dir", local])  # default featurizers: datamol (+ morgan)
    assert result.output.count("xai4chem · setup") == 2
    library = ReferenceLibrary(local)
    assert library.featurizers == ["morgan", "datamol"] and not (local / "accfg.h5").exists()
    assert (local / "morgan_bits_indices.npy").exists()
    original = ReferenceLibrary(lib)
    np.testing.assert_array_equal(library.matrix("datamol", [4, 2]), original.matrix("datamol", [4, 2]))
    run(["setup", "--library-dir", local, "-f", "accfg"])  # adds what is missing
    assert ReferenceLibrary(local).has("accfg")
    # A library from an older xai4chem is replaced, and only its own files go
    old = tmp_path / "old"
    old.mkdir()
    (old / "library.json").write_text(json.dumps({"schema": "xai4chem.library.v1", "name": "erl0", "features": {}}))
    (old / "notes.txt").write_text("mine")
    result = run(["setup", "--library-dir", old])
    assert "older xai4chem" in result.output and (old / "notes.txt").exists() and ReferenceLibrary(old).has("datamol")
    # A corrupted download is refused
    manifest = json.loads((remote / "library.json").read_text())
    manifest["files"]["panel.npy"]["sha256"] = "0" * 64
    (remote / "library.json").write_text(json.dumps(manifest))
    fails(["setup", "--library-dir", tmp_path / "other"], "checksum")
