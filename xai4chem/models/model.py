"""Loading a model directory, predicting, and merging soft and hard explanations.

For one molecule the gate weight ``a`` is a scalar, so attributions combine linearly and exactly:

- ``φ_S`` is TreeSHAP of S and sums to ``S(x) − E[S]``.
- ``φ_H`` is TreeSHAP of H in margin space and sums to ``h(x) − E[h]``.
- ``φ_HS = φ_H · (H_S(x) − H_S(E[h])) / (h(x) − E[h])`` carries H onto S's scale. This is the rescale
  rule (DeepLIFT) with E[h] as reference: the secant slope of the isotonic map, so a decreasing map
  flips the signs.
- ``φ_P = (1 − a)·φ_S + a·φ_HS`` sums to ``P(x) − [(1 − a)·E[S] + a·H_S(E[h])]``.

That decomposition is exact *given* ``a``. The gate itself is not attributed to features; T, ``a`` and
the nearest labelled analogue are reported alongside instead.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from rdkit import Chem, RDLogger

from xai4chem.data.library import BitDictionary
from xai4chem.data.similarity import binarize, fingerprints, nearest
from xai4chem.featurizers import featurizer_from_config
from xai4chem.models.hard import apply_isotonic, ramp
from xai4chem.models.surrogate import contributions, inverse, predict_margin
from xai4chem.progress import SILENT
from xai4chem.run import (
    BIT_DICTIONARY_NAME,
    GOLDEN_NAME,
    H_NAME,
    H_TO_S_NAME,
    HARD_META_NAME,
    LABELLED_FP_NAME,
    LABELLED_NAME,
    S_NAME,
    SOFT_META_NAME,
    column_dir,
    hard_dir,
    has_hard_head,
    read_json,
    read_manifest,
    write_json,
)

RDLogger.DisableLog("rdApp.*")

SLOPE_EPS = 1e-6
GOLDEN_TOL = 1e-4
BATCH = 20000
LIBRARY_THRESHOLD = 50000


def _load_booster(path):
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


@contextmanager
def _nothing():
    yield lambda done: None


def valid_mask(smiles):
    return np.array([Chem.MolFromSmiles(s) is not None for s in smiles], dtype=bool)


class HardHead(object):
    """The hard head of one column: H, its isotonic map onto S, and the gate's labelled set."""

    def __init__(self, hdir):
        self.meta = read_json(hdir / HARD_META_NAME)
        self.h = _load_booster(hdir / H_NAME)
        self.iso = read_json(hdir / H_TO_S_NAME)
        labelled = pd.read_csv(hdir / LABELLED_NAME)
        self.labelled_smiles = labelled["smiles"].tolist()
        self.labels = labelled["label"].to_numpy()
        fp_path = hdir / LABELLED_FP_NAME
        self.labelled_fps = sp.load_npz(fp_path) if fp_path.exists() else fingerprints(self.labelled_smiles)
        self.a_max = self.meta["a_max"]
        self.task = self.meta["task"]

    def gate(self, T):
        """The blend weight ``a = a_max · ramp(T)`` for similarities ``T``."""
        return self.a_max * ramp(T, self.meta["t_lo"], self.meta["t_hi"])


class Column(object):
    """One teacher column: S, its training metadata, and the hard head if one was learned."""

    def __init__(self, model_dir, entry):
        self.id = entry["id"]
        self.name = entry["name"]
        self.transform = entry["transform"]
        self.dir = column_dir(model_dir, self.id)
        self.soft_meta = read_json(self.dir / SOFT_META_NAME)
        self.s = _load_booster(self.dir / S_NAME)
        self.hard = HardHead(hard_dir(model_dir, self.id)) if has_hard_head(model_dir, self.id) else None


class XAIModel(object):
    """A trained xai4chem model: S per column, plus an optional hard head per column.

    Parameters
    ----------
    model_dir : str or Path
        A directory written by ``xai4chem fit``.
    check_golden : bool
        Re-score the stored golden molecules and refuse to load if this environment does not reproduce
        them. That, rather than a version pin, is the evidence that predictions here match training.
    """

    def __init__(self, model_dir, check_golden=True):
        self.dir = Path(model_dir)
        self.manifest = read_manifest(self.dir)
        self.featurizer = featurizer_from_config(self.manifest["featurizer"])
        if (self.dir / BIT_DICTIONARY_NAME).exists():
            self.featurizer.bit_dictionary = BitDictionary.load(self.dir / BIT_DICTIONARY_NAME)
        self.columns = [Column(self.dir, c) for c in self.manifest["columns"]]
        if check_golden and (self.dir / GOLDEN_NAME).exists():
            self._check_golden()

    @property
    def column_names(self):
        """Names of the teacher columns the model was trained on."""
        return [c.name for c in self.columns]

    @property
    def sparse(self):
        """Whether the model's features are sparse counts (zeros treated as missing)."""
        return self.featurizer.sparse

    def column(self, name):
        """The teacher column called ``name``.

        Parameters
        ----------
        name : str
            A name from :attr:`column_names`.

        Returns
        -------
        Column
            Its S, training metadata and hard head (or None).
        """
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"No column {name!r}. Columns: {self.column_names}")

    def _select(self, columns):
        if columns is None:
            return self.columns
        unknown = [c for c in columns if c not in self.column_names]
        if unknown:
            raise KeyError(f"Unknown columns {unknown}. Columns: {self.column_names}")
        return [c for c in self.columns if c.name in columns]

    def featurize(self, smiles):
        """Raw features of ``smiles`` with the model's featurizer.

        Parameters
        ----------
        smiles : sequence of str
            Molecules; unparseable ones get a row of zeros.

        Returns
        -------
        numpy.ndarray
            One row per molecule, one column per feature.
        """
        return self.featurizer.transform_raw(list(smiles))

    def _fingerprints(self, smiles, X):
        return binarize(X) if self.featurizer.name == "morgan" else fingerprints(smiles)

    def explain(self, smiles, X=None, columns=None):
        """Predictions and exact attributions for every requested column, on the model's scale.

        Attribution arrays are dense (``n × features``), so explain hundreds or thousands of molecules at a
        time, not millions; :meth:`predict` skips attributions.

        Parameters
        ----------
        smiles : sequence of str
            Molecules to explain.
        X : array-like, optional
            Their features, if already computed with :meth:`featurize`.
        columns : list of str, optional
            Teacher columns to use (default: all).

        Returns
        -------
        dict
            ``{column_name: dict}`` of arrays over molecules: ``S``, ``phi_S``, ``base_S``, ``P``, ``phi_P``,
            ``base_P`` and ``prob`` (P on the teacher's scale). Columns with a hard head also get ``h``,
            ``phi_H``, ``base_H``, ``HS``, ``phi_HS``, ``base_HS``, ``T``, ``a``, ``nearest`` and ``flat``
            (molecules whose H attribution could not be rescaled).
        """
        smiles = list(smiles)
        X = self.featurize(smiles) if X is None else X
        fps = None
        out = {}
        for col in self._select(columns):
            c = contributions(col.s, X, self.sparse)
            r = {"phi_S": c[:, :-1], "base_S": c[:, -1], "S": c.sum(1)}
            if col.hard is not None:
                hh = col.hard
                ch = contributions(hh.h, X, self.sparse)
                phi_H, base_H = ch[:, :-1], ch[:, -1]
                h = ch.sum(1)
                HS = apply_isotonic(h, hh.iso)
                base_HS = apply_isotonic(base_H, hh.iso)
                den = h - base_H
                flat = np.abs(den) < SLOPE_EPS
                slope = np.where(flat, 0.0, (HS - base_HS) / np.where(flat, 1.0, den))
                phi_HS = phi_H * slope[:, None]
                if fps is None:
                    fps = self._fingerprints(smiles, X)
                T, near = nearest(fps, hh.labelled_fps)
                a = hh.gate(T)
                r.update(
                    {
                        "h": h,
                        "phi_H": phi_H,
                        "base_H": base_H,
                        "HS": HS,
                        "phi_HS": phi_HS,
                        "base_HS": base_HS,
                        "flat": flat,
                        "T": T,
                        "a": a,
                        "nearest": near,
                        "P": (1 - a) * r["S"] + a * HS,
                        "phi_P": (1 - a)[:, None] * r["phi_S"] + a[:, None] * phi_HS,
                        "base_P": (1 - a) * r["base_S"] + a * base_HS,
                    }
                )
            else:
                r.update({"P": r["S"], "phi_P": r["phi_S"], "base_P": r["base_S"]})
            r["prob"] = inverse(r["P"], col.transform)
            out[col.name] = r
        return out

    def _margins_batch(self, smiles, columns, X=None, fps=None):
        X = self.featurize(smiles) if X is None else X
        out = {}
        for col in columns:
            S = predict_margin(col.s, X, self.sparse)
            r = {"S": S, "P": S}
            if col.hard is not None:
                hh = col.hard
                HS = apply_isotonic(predict_margin(hh.h, X, self.sparse), hh.iso)
                if fps is None:
                    fps = self._fingerprints(smiles, X)
                T, _ = nearest(fps, hh.labelled_fps)
                a = hh.gate(T)
                r.update({"HS": HS, "T": T, "a": a, "P": (1 - a) * S + a * HS})
            out[col.name] = r
        return out

    def _library(self):
        from xai4chem.data.library import library_for_model

        try:
            return library_for_model(self.manifest)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return None

    def features(self, smiles, fingerprints_too=False, display=SILENT):
        """Features and validity for many molecules.

        Inputs of 50,000 molecules or more reuse the reference library: molecules found there (exact or
        canonical SMILES) get their precomputed rows and only the rest are featurized.

        Parameters
        ----------
        smiles : sequence of str
            Molecules.
        fingerprints_too : bool
            Also return binary Morgan fingerprints (for the gate) when the library can provide them.
        display : xai4chem.progress.Display
            Where progress goes (silent by default).

        Returns
        -------
        tuple
            ``(X, valid, fps)``: the features, which SMILES parse, and the fingerprints or None.
        """
        from xai4chem.data.features import align_to_library, featurize, load_features

        smiles = list(smiles)
        library = self._library() if len(smiles) >= LIBRARY_THRESHOLD else None
        same = (
            library is not None
            and library.has(self.featurizer.name)
            and library.manifest["features"][self.featurizer.name]["config"] == self.featurizer.to_config()
        )
        if not same:
            return featurize(self.featurizer, smiles, display), valid_mask(smiles), None
        rows = align_to_library(smiles, library)
        X = load_features(smiles, rows, library, self.featurizer, display)
        valid = rows >= 0
        missing = np.where(~valid)[0]
        valid[missing] = valid_mask([smiles[i] for i in missing])
        fps = None
        if fingerprints_too:
            if self.featurizer.name == "morgan":
                fps = binarize(X)
            elif library.has("morgan"):
                fps = binarize(load_features(smiles, rows, library, library.featurizer("morgan"), display))
        return X, valid, fps

    def scores(self, smiles, columns=None, batch_size=BATCH, display=SILENT):
        """S and P (and H_S, T, a) per column on the model's scale, without attributions.

        Parameters
        ----------
        smiles : sequence of str
            Molecules to score.
        columns : list of str, optional
            Teacher columns to use (default: all).
        batch_size : int
            Molecules scored at a time.
        display : xai4chem.progress.Display
            Where progress goes (silent by default).

        Returns
        -------
        dict
            ``{column_name: dict}`` of arrays over molecules, plus ``"_valid"``: which SMILES parse.
        """
        smiles = list(smiles)
        cols = self._select(columns)
        need_fps = any(c.hard is not None for c in cols) and self.featurizer.name != "morgan"
        X, valid, fps = self.features(smiles, fingerprints_too=need_fps, display=display)
        parts = []
        with display.loop("scoring", len(smiles)) if len(smiles) > batch_size else _nothing() as tick:
            for start in range(0, len(smiles), batch_size):
                stop = min(start + batch_size, len(smiles))
                batch_fps = fps[start:stop] if fps is not None else None
                parts.append(self._margins_batch(smiles[start:stop], cols, X=X[start:stop], fps=batch_fps))
                tick(stop)
        out = (
            {c.name: {k: np.concatenate([p[c.name][k] for p in parts]) for k in parts[0][c.name]} for c in cols}
            if parts
            else {}
        )
        out["_valid"] = valid
        return out

    def predict(self, smiles, columns=None, details=False, batch_size=BATCH, display=SILENT):
        """One row per molecule, with P on the teacher's scale per column; cheap enough for millions.

        Parameters
        ----------
        smiles : sequence of str
            Molecules to score.
        columns : list of str, optional
            Teacher columns to use (default: all).
        details : bool
            Also give S and H_S (on the teacher's scale), T and ``a``.
        batch_size : int
            Molecules scored at a time.
        display : xai4chem.progress.Display
            Where progress goes (silent by default).

        Returns
        -------
        pandas.DataFrame
            ``smiles``, ``valid``, then the predictions (NaN for unparseable SMILES).
        """
        smiles = list(smiles)
        res = self.scores(smiles, columns=columns, batch_size=batch_size, display=display)
        df = pd.DataFrame({"smiles": smiles})
        for col in self._select(columns):
            r = res[col.name]
            df[col.name] = inverse(r["P"], col.transform)
            if details:
                df[f"{col.name}_S"] = inverse(r["S"], col.transform)
                if "HS" in r:
                    df[f"{col.name}_HS"] = inverse(r["HS"], col.transform)
                    df[f"{col.name}_T"] = r["T"]
                    df[f"{col.name}_a"] = r["a"]
        df.loc[~res["_valid"], df.columns[1:]] = np.nan
        df.insert(1, "valid", res["_valid"])
        return df

    def library_neighbours(self, smiles):
        """Each molecule's nearest neighbour in the reference library, and what the teacher said about it.

        This is the applicability domain of S: T close to 1 means S was trained on (nearly) this molecule;
        below ~0.3 it is extrapolating.

        Parameters
        ----------
        smiles : sequence of str
            Molecules to look up.

        Returns
        -------
        dict or None
            ``T``, ``row``, ``smiles`` and ``teacher`` (per column, on the teacher's scale; NaN where the soft
            labels did not cover the neighbour); None when the library or its Morgan matrix is missing.
        """
        import h5py

        from xai4chem.data.similarity import nearest_large
        from xai4chem.run import TARGETS_NAME

        library = self._library()
        if library is None or not library.has("morgan"):
            return None
        if getattr(self, "_library_fps", None) is None:
            self._library_fps = library.morgan_bits()
        T, rows = nearest_large(fingerprints(smiles), self._library_fps)
        valid = valid_mask(smiles)
        T = np.where(valid, T, np.nan)
        rows = np.where(valid, rows, -1)
        teacher = {}
        targets = self.dir / TARGETS_NAME
        if targets.exists():
            with h5py.File(targets, "r") as f:
                target_rows, Z = f["rows"][:], f["Z"][:]
            for col in self.columns:
                j = int(col.id[1:])  # Z has one column per soft-label column, skipped ones included
                by_row = np.full(len(library), np.nan, dtype=np.float32)
                ok = target_rows >= 0
                by_row[target_rows[ok]] = Z[ok, j]
                teacher[col.name] = np.where(valid, inverse(by_row[rows], col.transform), np.nan)
        neighbours = [library.smiles[i] if i >= 0 else None for i in rows]
        return {"T": T, "row": rows, "smiles": neighbours, "teacher": teacher}

    def _check_golden(self):
        golden = read_json(self.dir / GOLDEN_NAME)
        res = self.scores(golden["smiles"])
        for name, expected in golden["columns"].items():
            got = res[name]["P"]
            if not np.allclose(got, expected["P"], atol=GOLDEN_TOL):
                raise RuntimeError(
                    f"Golden check failed for column {name!r}: this environment does not reproduce the model "
                    f"(max deviation {np.max(np.abs(got - np.array(expected['P']))):.2e})."
                )


def write_golden(model_dir):
    model = XAIModel(model_dir, check_golden=False)
    smiles = model.manifest["golden_smiles"]
    res = model.scores(smiles)
    res.pop("_valid")
    write_json(
        Path(model_dir) / GOLDEN_NAME,
        {"smiles": smiles, "columns": {name: {"S": r["S"].tolist(), "P": r["P"].tolist()} for name, r in res.items()}},
    )
