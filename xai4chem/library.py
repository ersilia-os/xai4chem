"""The reference library and everything precomputed over it.

Featurizing ~1.36M molecules is the expensive, model-independent part of the pipeline, so it happens
once (``xai4chem setup``) and is stored in a library directory (default ``~/.xai4chem/library/``)::

    library.json          the manifest: where the SMILES and every feature matrix live
    <featurizer>.h5       dataset ``X`` (rows x features, raw dtype), uncompressed for fast reads
    panel.npy             row indices of a fixed random subset, used for global explanations
    bit_dictionary.json   per Morgan bit, the substructures that set it across the library

By default the library is **Ersilia's reference library**, the same 1,355,109 molecules Olinda
distills over. Olinda publishes it with its Morgan counts precomputed (``erl0_qmfp_r3.h5``), and
xai4chem's Morgan featurizer is identical to Olinda's, so that file is used in place, not copied:
``library.json`` points at it. Soft labels made for Olinda (teacher values in library order) work
unchanged.

A matrix entry in the manifest names its file and dataset, so a matrix can live outside the library
directory. Row order is always the order of the SMILES.
"""

import json
import os
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np

from xai4chem.console import echo, filesize, live_status, status_line, success, sweep_progress
from xai4chem.console import path as cpath
from xai4chem.featurizers import featurizer_from_config, get_featurizer
from xai4chem.featurizers.morgan import COUNT_CLIP, NBITS, RADIUS

SCHEMA = "xai4chem.library.v1"
LIBRARY_JSON = "library.json"
PANEL_NAME = "panel.npy"
BIT_DICTIONARY_NAME = "bit_dictionary.json"
PANEL_SIZE = 50000
BIT_DICTIONARY_MOLS = 100000
DATAMOL_FIT_MOLS = 50000
TOP_FRAGMENTS = 10
FEATURIZE_CHUNK = 5000
READ_CHUNK = 100000

ERSILIA_LIBRARY_NAME = "erl0"
ERSILIA_LIBRARY_FILENAME = "erl0_qmfp_r3.h5"
ERSILIA_LIBRARY_URL = f"https://eosvc-public.s3.eu-central-1.amazonaws.com/olinda/data/{ERSILIA_LIBRARY_FILENAME}"
OLINDA_HOME = Path.home() / ".olinda"


def home_dir():
    return Path(os.environ.get("XAI4CHEM_HOME", Path.home() / ".xai4chem"))


def default_library_dir():
    return home_dir() / "library"


def default_n_jobs():
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# --- Ersilia reference library ---------------------------------------------------------------------


def _download(url, dest):
    """Download ``url`` to ``dest`` atomically, with a live progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    started = time.time()
    try:
        with live_status() as update:

            def hook(blocks, block_size, total):
                done = blocks * block_size
                frac = min(1.0, done / total) if total and total > 0 else None
                rate = done / max(time.time() - started, 1e-6)
                update(status_line("downloading", frac=frac, pairs=[("", filesize(done))], tail=f"{filesize(rate)}/s"))

            urlretrieve(url, tmp, reporthook=hook)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def locate_ersilia_library(download=True):
    """Path to ``erl0_qmfp_r3.h5``: Olinda's copy if there is one, else downloaded once into ~/.xai4chem."""
    for candidate in (OLINDA_HOME / ERSILIA_LIBRARY_FILENAME, home_dir() / ERSILIA_LIBRARY_FILENAME):
        if candidate.exists():
            echo(f"Ersilia reference library found · [dim]{cpath(candidate)}[/]", "info")
            return candidate
    if not download:
        raise FileNotFoundError(f"{ERSILIA_LIBRARY_FILENAME} not found in {OLINDA_HOME} or {home_dir()}")
    dest = home_dir() / ERSILIA_LIBRARY_FILENAME
    echo(f"downloading the Ersilia reference library (2.9 GB) → [dim]{cpath(dest)}[/]", "run")
    _download(ERSILIA_LIBRARY_URL, dest)
    success(f"downloaded → [dim]{cpath(dest)}[/]")
    return dest


def _check_qmfp(path):
    """Refuse an Olinda library whose Morgan counts differ from xai4chem's."""
    with h5py.File(path, "r") as f:
        spec = json.loads(f.attrs.get("fingerprint", "{}"))
        layout = f.attrs.get("layout", "")
        n = f["data"].shape[0]
    morgan = spec.get("morgan", {})
    ok = (
        spec.get("type") == "qmfp"
        and int(spec.get("count_clip", -1)) == COUNT_CLIP
        and int(morgan.get("radius", -1)) == RADIUS
        and int(morgan.get("nBits", -1)) == NBITS
        and layout == "decoded_uint8"
    )
    if not ok:
        raise ValueError(f"{path} does not hold Morgan r{RADIUS}/{NBITS} counts clipped at {COUNT_CLIP} (decoded).")
    return n


# --- computing --------------------------------------------------------------------------------------


def _featurize_chunk(args):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    cfg, smiles = args
    return featurizer_from_config(cfg).transform_raw(smiles)


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _imap(func, iterable, n_jobs):
    """Ordered map, in worker processes when ``n_jobs > 1`` (spawned, so it is safe on macOS)."""
    if n_jobs <= 1:
        yield from map(func, iterable)
        return
    with get_context("spawn").Pool(n_jobs) as pool:
        yield from pool.imap(func, iterable)


def write_matrix(path, featurizer, smiles, n_jobs=1):
    """Featurize ``smiles`` into ``path`` (dataset ``X``), streaming chunks as workers finish them."""
    cfg = featurizer.to_config()
    dtype = np.float32 if featurizer.name == "datamol" else np.uint8
    tmp = Path(str(path) + ".tmp")
    with h5py.File(tmp, "w") as f:
        ds = f.create_dataset("X", shape=(len(smiles), len(featurizer.feature_names)), dtype=dtype)
        f.attrs["config"] = json.dumps(cfg)
        i = 0
        with sweep_progress(f"featurizing ({featurizer.name})", len(smiles)) as tick:
            for block in _imap(_featurize_chunk, ((cfg, c) for c in _chunks(smiles, FEATURIZE_CHUNK)), n_jobs):
                ds[i : i + len(block)] = block
                i += len(block)
                tick(i)
    tmp.replace(path)


def _fragment(mol, center, radius):
    from rdkit import Chem

    if radius == 0:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=[center])
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
    if not bonds:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=[center])
    atoms = set()
    for b in bonds:
        bond = mol.GetBondWithIdx(b)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=bonds)


def _bit_fragments_chunk(smiles):
    from collections import Counter

    from rdkit import Chem, RDLogger

    from xai4chem.featurizers.morgan import morgan_bit_info

    RDLogger.DisableLog("rdApp.*")
    counts = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for bit, occ in morgan_bit_info(mol).items():
            c = counts.setdefault(bit, Counter())
            for center, radius in occ:
                c[(radius, _fragment(mol, center, radius))] += 1
    return counts


def compute_bit_dictionary(smiles, n_jobs=1):
    """Name every Morgan bit by the substructures that set it, counted over ``smiles``."""
    from collections import Counter

    total = [Counter() for _ in range(NBITS)]
    done = 0
    chunk = 2000
    with sweep_progress("naming Morgan bits", len(smiles)) as tick:
        for counts in _imap(_bit_fragments_chunk, _chunks(smiles, chunk), n_jobs):
            for bit, c in counts.items():
                total[bit].update(c)
            done += chunk
            tick(min(done, len(smiles)))
    out = {}
    for bit in range(NBITS):
        n = sum(total[bit].values())
        out[str(bit)] = {
            "occurrences": n,
            "distinct": len(total[bit]),
            "fragments": [
                {"smiles": frag, "radius": radius, "share": c / n}
                for (radius, frag), c in total[bit].most_common(TOP_FRAGMENTS)
            ],
        }
    return {"n_molecules": len(smiles), "bits": out}


def setup_library(
    source="ersilia",
    featurizers=("morgan",),
    library_dir=None,
    n_jobs=None,
    panel_size=PANEL_SIZE,
    bit_dictionary_mols=BIT_DICTIONARY_MOLS,
    force=False,
    seed=42,
):
    """Build (or extend) a reference library directory. Anything already present is kept unless ``force``."""
    from rdkit import Chem, RDLogger

    from xai4chem.io import read_smiles

    RDLogger.DisableLog("rdApp.*")
    library_dir = Path(library_dir or default_library_dir())
    library_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = n_jobs or default_n_jobs()
    manifest_path = library_dir / LIBRARY_JSON
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    if manifest is None or force:
        if source == "ersilia":
            h5 = locate_ersilia_library()
            n = _check_qmfp(h5)
            manifest = {
                "name": ERSILIA_LIBRARY_NAME,
                "n_molecules": int(n),
                "smiles": {"file": str(h5), "dataset": "input"},
                "features": {
                    "morgan": {"file": str(h5), "dataset": "data", "config": get_featurizer("morgan").to_config()}
                },
            }
        else:
            echo(f"reading SMILES · [dim]{cpath(source)}[/]", "run")
            smiles = read_smiles(source)
            valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
            if len(valid) < len(smiles):
                echo(f"dropped {len(smiles) - len(valid):,} unparseable SMILES", "warning")
            smiles = list(dict.fromkeys(valid))
            with h5py.File(library_dir / "smiles.h5", "w") as f:
                f.create_dataset("input", data=np.array(smiles, dtype=object), dtype=h5py.string_dtype())
            manifest = {
                "name": Path(source).stem,
                "n_molecules": len(smiles),
                "smiles": {"file": str(library_dir / "smiles.h5"), "dataset": "input"},
                "features": {},
            }
        manifest["schema"] = SCHEMA
        manifest["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_manifest(library_dir, manifest)
        success(
            f"library [bold]{manifest['name']}[/] · {manifest['n_molecules']:,} molecules → [dim]{cpath(library_dir)}[/]"
        )

    library = ReferenceLibrary(library_dir)
    rng = np.random.default_rng(seed)
    if force or not (library_dir / PANEL_NAME).exists():
        panel = np.sort(rng.choice(len(library), size=min(panel_size, len(library)), replace=False))
        np.save(library_dir / PANEL_NAME, panel)
        success(f"explanation panel · {len(panel):,} molecules")

    for name in featurizers:
        if name in manifest["features"] and not force:
            echo(f"{name} features already present · [dim]{cpath(manifest['features'][name]['file'])}[/]", "info")
            continue
        featurizer = get_featurizer(name)
        smiles = library.smiles
        if name == "datamol":
            idx = rng.choice(len(smiles), size=min(DATAMOL_FIT_MOLS, len(smiles)), replace=False)
            echo(f"fitting datamol descriptor filters on {len(idx):,} molecules", "run")
            featurizer.fit([smiles[i] for i in idx])
        echo(f"featurizing {len(smiles):,} molecules with [bold]{name}[/] · {n_jobs} workers", "run")
        out = library_dir / f"{name}.h5"
        write_matrix(out, featurizer, smiles, n_jobs=n_jobs)
        manifest["features"][name] = {"file": str(out), "dataset": "X", "config": featurizer.to_config()}
        _write_manifest(library_dir, manifest)
        success(f"{name} → [dim]{cpath(out)}[/] · {filesize(out.stat().st_size)}")

    if "morgan" in manifest["features"] and (force or not (library_dir / BIT_DICTIONARY_NAME).exists()):
        smiles = library.smiles
        idx = np.sort(rng.choice(len(smiles), size=min(bit_dictionary_mols, len(smiles)), replace=False))
        echo(f"building the Morgan bit dictionary over {len(idx):,} molecules · {n_jobs} workers", "run")
        bd = compute_bit_dictionary([smiles[i] for i in idx], n_jobs=n_jobs)
        (library_dir / BIT_DICTIONARY_NAME).write_text(json.dumps(bd))
        success(f"bit dictionary → [dim]{cpath(library_dir / BIT_DICTIONARY_NAME)}[/]")
    return library_dir


def _write_manifest(library_dir, manifest):
    (Path(library_dir) / LIBRARY_JSON).write_text(json.dumps(manifest, indent=2))


# --- reading ----------------------------------------------------------------------------------------


class BitDictionary(object):
    """Names folded Morgan bits by the substructures that set them across the library."""

    def __init__(self, data):
        self.data = data
        self.bits = data["bits"]

    @classmethod
    def load(cls, path):
        """Read a bit dictionary written by ``xai4chem setup``."""
        with open(path) as f:
            return cls(json.load(f))

    def fragments(self, bit):
        """The most common substructures that set ``bit``, with their shares."""
        return self.bits[str(bit)]["fragments"]

    def collision(self, bit):
        """Share of the bit's occurrences not explained by its most common substructure."""
        frags = self.fragments(bit)
        return 1.0 - frags[0]["share"] if frags else float("nan")

    def label(self, bit):
        """A short name for ``bit``: its most common substructure and that substructure's share."""
        frags = self.fragments(bit)
        if not frags:
            return f"bit {bit}"
        top = frags[0]
        return f"bit {bit}: {top['smiles']} ({top['share']:.0%})"


class ReferenceLibrary(object):
    """Read access to a library directory written by :func:`setup_library`."""

    def __init__(self, library_dir=None):
        self.dir = Path(library_dir or default_library_dir())
        path = self.dir / LIBRARY_JSON
        if not path.exists():
            raise FileNotFoundError(f"No reference library at {self.dir}. Run `xai4chem setup` first.")
        self.manifest = json.loads(path.read_text())
        self._smiles = None
        self._index = None

    @property
    def name(self):
        """The library's name (``erl0`` for Ersilia's reference library)."""
        return self.manifest["name"]

    @property
    def featurizers(self):
        """Names of the featurizers precomputed over the library."""
        return list(self.manifest["features"])

    @property
    def smiles(self):
        """The library SMILES, in row order (loaded once, on first access)."""
        if self._smiles is None:
            src = self.manifest["smiles"]
            with h5py.File(src["file"], "r") as f:
                raw = f[src["dataset"]][:]
            self._smiles = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
        return self._smiles

    @property
    def index(self):
        """SMILES → row, for aligning soft labels to the library."""
        if self._index is None:
            self._index = {s: i for i, s in enumerate(self.smiles)}
        return self._index

    def __len__(self):
        return int(self.manifest["n_molecules"])

    def has(self, featurizer_name):
        """Whether the library has precomputed features for ``featurizer_name``."""
        return featurizer_name in self.manifest["features"]

    def _source(self, name):
        if not self.has(name):
            raise FileNotFoundError(
                f"The library at {self.dir} has no {name} features. Run `xai4chem setup --featurizer {name}`."
            )
        return self.manifest["features"][name]

    def featurizer(self, name):
        """The featurizer behind a library matrix (Morgan comes with its bit dictionary)."""
        featurizer = featurizer_from_config(self._source(name)["config"])
        if name == "morgan" and (self.dir / BIT_DICTIONARY_NAME).exists():
            featurizer.bit_dictionary = self.bit_dictionary()
        return featurizer

    def bit_dictionary(self):
        """The Morgan bit dictionary computed at setup."""
        return BitDictionary.load(self.dir / BIT_DICTIONARY_NAME)

    def matrix(self, name, rows=None):
        """Raw feature rows, in the order given by ``rows`` (all rows if ``None``)."""
        src = self._source(name)
        with h5py.File(src["file"], "r") as f:
            ds = f[src["dataset"]]
            if rows is None:
                return ds[:]
            rows = np.asarray(rows)
            if len(rows) > 0.05 * ds.shape[0]:
                # Point selection is slow for large row sets; contiguous chunk reads are not
                return self._gather(ds, rows)
            order = np.argsort(rows)
            out = np.empty((len(rows), ds.shape[1]), dtype=ds.dtype)
            out[order] = ds[rows[order]]
            return out

    @staticmethod
    def _gather(ds, rows):
        out = np.empty((len(rows), ds.shape[1]), dtype=ds.dtype)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        pos = 0
        for start in range(0, ds.shape[0], READ_CHUNK):
            stop = min(start + READ_CHUNK, ds.shape[0])
            end = np.searchsorted(sorted_rows, stop, side="left")
            if end > pos:
                block = ds[start:stop]
                out[order[pos:end]] = block[sorted_rows[pos:end] - start]
                pos = end
        return out

    def csr(self, name, rows=None, verb=None):
        """Rows as a float32 CSR matrix, read in chunks so a dense copy of the library never exists.

        Only meaningful for featurizers whose zeros mean "absent" (Morgan, AccFG): XGBoost treats the
        implicit zeros of a CSR matrix as missing, which is the same split for a count feature and lets
        training run on ~3% of the data.
        """
        import scipy.sparse as sp

        src = self._source(name)
        with h5py.File(src["file"], "r") as f:
            ds = f[src["dataset"]]
            n = ds.shape[0]
            rows = np.arange(n) if rows is None else np.asarray(rows)
            order = np.argsort(rows, kind="stable")
            sorted_rows = rows[order]
            blocks, pos = [], 0
            with sweep_progress(verb or f"reading {name} features", len(rows)) as tick:
                for start in range(0, n, READ_CHUNK):
                    stop = min(start + READ_CHUNK, n)
                    end = np.searchsorted(sorted_rows, stop, side="left")
                    if end > pos:
                        block = ds[start:stop][sorted_rows[pos:end] - start]
                        blocks.append(sp.csr_matrix(block, dtype=np.float32))
                        pos = end
                        tick(pos)
            width = ds.shape[1]
        X = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, width), dtype=np.float32)
        if np.all(order[:-1] < order[1:]):
            return X
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return X[inverse]

    def panel(self):
        """Row indices of the explanation panel."""
        return np.load(self.dir / PANEL_NAME)
