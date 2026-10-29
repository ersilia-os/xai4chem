"""Getting a reference library onto disk (``xai4chem setup``), and building the published one.

Ersilia's reference library (``erl0``, 1,355,109 molecules) is published ready-made: SMILES, every
featurizer's matrix, the explanation panel and the Morgan bit names. ``setup`` downloads the files it
needs into the library directory, checks them against the published checksums, and caches the binary
Morgan fingerprints locally. Set ``XAI4CHEM_LIBRARY_URL`` to download from elsewhere (a mirror or a local
folder).

A library of your own SMILES (``setup --library my.csv``) is built locally instead, with the same
functions that build the published one (:func:`build_library`): sequential, resumable, run once.
"""

import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import h5py
import numpy as np

from xai4chem.data.library import (
    BIT_DICTIONARY_NAME,
    ERSILIA_LIBRARY_NAME,
    LIBRARY_JSON,
    MORGAN_BITS_INDICES,
    MORGAN_BITS_INDPTR,
    PANEL_NAME,
    SCHEMA,
    SMILES_NAME,
    ReferenceLibrary,
    default_library_dir,
    write_manifest,
)
from xai4chem.featurizers import FEATURIZERS, get_featurizer
from xai4chem.featurizers.morgan import NBITS, fragment_smiles, morgan_bit_info
from xai4chem.progress import SILENT

ERSILIA_LIBRARY_URL = "https://eosvc-public.s3.eu-central-1.amazonaws.com/xai4chem/data/erl0"
PANEL_SIZE = 50_000
BIT_DICTIONARY_MOLS = 20_000
DATAMOL_FIT_MOLS = 20_000
TOP_FRAGMENTS = 10
CHUNK = 5_000
DOWNLOAD_BLOCK = 1 << 20


def library_url():
    """Where the published library lives (``XAI4CHEM_LIBRARY_URL`` overrides; a local folder works too)."""
    url = os.environ.get("XAI4CHEM_LIBRARY_URL", ERSILIA_LIBRARY_URL).rstrip("/")
    return url if urlparse(url).scheme else Path(url).resolve().as_uri()


# --- writing matrices -------------------------------------------------------------------------------


def write_csr(path, X, config):
    """A CSR matrix of counts as HDF5 (``data`` uint8, ``indices`` int16, ``indptr`` int64)."""
    tmp = Path(str(path) + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("data", data=X.data.astype(np.uint8))
        f.create_dataset("indices", data=X.indices.astype(np.int16))
        f.create_dataset("indptr", data=X.indptr.astype(np.int64))
        f.attrs["shape"] = X.shape
        f.attrs["config"] = json.dumps(config)
    tmp.replace(path)


def write_dense(path, X, config):
    """A dense descriptor matrix as HDF5 (dataset ``X``, float32)."""
    tmp = Path(str(path) + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("X", data=np.asarray(X, dtype=np.float32))
        f.attrs["config"] = json.dumps(config)
    tmp.replace(path)


def featurize_library(featurizer, smiles, display=SILENT):
    """Featurize ``smiles`` chunk by chunk: CSR for count featurizers, a dense float32 array otherwise."""
    import scipy.sparse as sp

    blocks = []
    with display.loop(f"featurizing ({featurizer.name})", len(smiles)) as tick:
        for start in range(0, len(smiles), CHUNK):
            block = featurizer.transform_raw(smiles[start : start + CHUNK])
            blocks.append(sp.csr_matrix(block, dtype=np.float32) if featurizer.sparse else block.astype(np.float32))
            tick(start + len(block))
    return sp.vstack(blocks, format="csr") if featurizer.sparse else np.vstack(blocks)


def write_features(library_dir, featurizer, X):
    """Write a featurizer's matrix into the library directory; return its ``library.json`` entry."""
    name = f"{featurizer.name}.h5"
    if featurizer.sparse:
        write_csr(Path(library_dir) / name, X, featurizer.to_config())
        return {"file": name, "format": "csr", "dtype": "uint8", "config": featurizer.to_config()}
    write_dense(Path(library_dir) / name, X, featurizer.to_config())
    return {"file": name, "format": "dense", "dtype": "float32", "config": featurizer.to_config()}


def _count_environments(mol, total):
    """Add every atom environment of ``mol`` to the per-bit counters, named by its SMILES.

    One SMILES per occurrence: caching by Morgan identifier is 7x faster but misnames ~1% of bits,
    because distinct environments (e.g. ring sizes) can share an identifier.
    """
    for bit, occ in morgan_bit_info(mol).items():
        for center, radius in occ:
            total[bit][(radius, fragment_smiles(mol, center, radius))] += 1


def compute_bit_dictionary(smiles, display=SILENT):
    """Name every Morgan bit by the substructures that set it, counted over ``smiles``."""
    from rdkit import Chem

    total = [Counter() for _ in range(NBITS)]
    with display.loop("naming Morgan bits", len(smiles)) as tick:
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                _count_environments(mol, total)
            if i % 500 == 0:
                tick(i)
        tick(len(smiles))
    bits = {}
    for bit in range(NBITS):
        n = sum(total[bit].values())
        bits[str(bit)] = {
            "occurrences": n,
            "distinct": len(total[bit]),
            "fragments": [
                {"smiles": frag, "radius": radius, "share": c / n}
                for (radius, frag), c in total[bit].most_common(TOP_FRAGMENTS)
            ],
        }
    return {"n_molecules": len(smiles), "bits": bits}


# --- building a library from SMILES -----------------------------------------------------------------


def _write_smiles(library_dir, smiles):
    with h5py.File(Path(library_dir) / SMILES_NAME, "w") as f:
        f.create_dataset("input", data=np.array(smiles, dtype=object), dtype=h5py.string_dtype())


def build_library(
    smiles, library_dir, name, featurizers=("morgan", "datamol"), source=None, reuse=None, seed=42, display=SILENT
):
    """Build (or complete) a library directory from SMILES; what is already there is kept.

    ``reuse`` maps featurizer names to ``(matrix, config)`` computed elsewhere for the same SMILES, in the
    same order (the published library's maintainer passes earlier matrices so they are not recomputed).
    Returns the directory.
    """
    library_dir = Path(library_dir)
    library_dir.mkdir(parents=True, exist_ok=True)
    path = library_dir / LIBRARY_JSON
    manifest = json.loads(path.read_text()) if path.exists() else None
    if manifest is None:
        _write_smiles(library_dir, smiles)
        manifest = {
            "schema": SCHEMA,
            "name": name,
            "n_molecules": len(smiles),
            "source": source,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "smiles": {"file": SMILES_NAME, "dataset": "input"},
            "features": {},
        }
        write_manifest(library_dir, manifest)
    rng = np.random.default_rng(seed)
    names = list(dict.fromkeys(["morgan", *featurizers]))
    steps = 4
    display.step(1, steps, "Features")
    for fname in names:
        if fname in manifest["features"]:
            display.note(f"{fname} · ready")
            continue
        started = time.time()
        if reuse and fname in reuse:
            X, config = reuse[fname]
            featurizer = FEATURIZERS[fname].from_config(config)
        else:
            featurizer = get_featurizer(fname)
            if fname == "datamol":
                idx = rng.choice(len(smiles), size=min(DATAMOL_FIT_MOLS, len(smiles)), replace=False)
                featurizer.fit([smiles[i] for i in idx])
            X = featurize_library(featurizer, smiles, display)
        manifest["features"][fname] = write_features(library_dir, featurizer, X)
        write_manifest(library_dir, manifest)
        display.note(f"{fname} · {time.time() - started:.0f} s", "success")

    display.step(2, steps, "Explanation panel")
    if not (library_dir / PANEL_NAME).exists():
        panel = np.sort(rng.choice(len(smiles), size=min(PANEL_SIZE, len(smiles)), replace=False))
        np.save(library_dir / PANEL_NAME, panel)
    display.note(f"{len(np.load(library_dir / PANEL_NAME)):,} molecules")

    display.step(3, steps, "Morgan bit names")
    if not (library_dir / BIT_DICTIONARY_NAME).exists():
        idx = np.sort(rng.choice(len(smiles), size=min(BIT_DICTIONARY_MOLS, len(smiles)), replace=False))
        bd = compute_bit_dictionary([smiles[i] for i in idx], display)
        (library_dir / BIT_DICTIONARY_NAME).write_text(json.dumps(bd))
    n_named = json.loads((library_dir / BIT_DICTIONARY_NAME).read_text())["n_molecules"]
    display.note(f"from {n_named:,} molecules")

    display.step(4, steps, "Fingerprint cache")
    _fingerprint_cache(library_dir)
    return library_dir


def _fingerprint_cache(library_dir):
    if not (Path(library_dir) / MORGAN_BITS_INDICES).exists() or not (Path(library_dir) / MORGAN_BITS_INDPTR).exists():
        ReferenceLibrary(library_dir).morgan_bits()


# --- the published library --------------------------------------------------------------------------


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(DOWNLOAD_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def publish_manifest(library_dir):
    """Add sizes and checksums of every file to ``library.json``, ready to upload the directory as is."""
    library_dir = Path(library_dir)
    manifest = json.loads((library_dir / LIBRARY_JSON).read_text())
    files = [manifest["smiles"]["file"], PANEL_NAME, BIT_DICTIONARY_NAME]
    files += [e["file"] for e in manifest["features"].values()]
    manifest["files"] = {
        f: {"bytes": (library_dir / f).stat().st_size, "sha256": _sha256(library_dir / f)} for f in files
    }
    write_manifest(library_dir, manifest)
    return manifest


def _fetch(url, dest, size, display, label):
    tmp = Path(str(dest) + ".tmp")
    try:
        with urlopen(url) as response, open(tmp, "wb") as out:
            with display.loop(f"downloading {label} (MB)", max(1, size // 1_000_000)) as tick:
                done = 0
                for block in iter(lambda: response.read(DOWNLOAD_BLOCK), b""):
                    out.write(block)
                    done += len(block)
                    tick(done // 1_000_000)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def download_library(library_dir, featurizers, display=SILENT):
    """Download the published library's files that ``library_dir`` lacks; return the directory."""
    library_dir = Path(library_dir)
    library_dir.mkdir(parents=True, exist_ok=True)
    base = library_url()
    try:
        with urlopen(f"{base}/{LIBRARY_JSON}") as response:
            remote = json.loads(response.read())
    except OSError as exc:  # urllib's HTTPError and URLError are OSErrors
        raise FileNotFoundError(
            f"Could not reach the Ersilia reference library at {base} ({exc}). Check the connection, set "
            "XAI4CHEM_LIBRARY_URL to a mirror, or build a library from your own SMILES with "
            "`xai4chem setup --library my_smiles.csv`."
        ) from exc
    names = list(dict.fromkeys(["morgan", *featurizers]))
    missing = [n for n in names if n not in remote["features"]]
    if missing:
        raise ValueError(f"The published library has no {', '.join(missing)} features.")
    local_path = library_dir / LIBRARY_JSON
    local = json.loads(local_path.read_text()) if local_path.exists() else None
    if local is None:
        local = {k: v for k, v in remote.items() if k not in ("features", "files")}
        local["features"] = {}
    wanted = [remote["smiles"]["file"], PANEL_NAME, BIT_DICTIONARY_NAME] + [
        remote["features"][n]["file"] for n in names
    ]
    todo = [f for f in wanted if not (library_dir / f).exists()]
    total = sum(remote["files"][f]["bytes"] for f in todo)
    if todo:
        display.note(f"{len(todo)} files · {total / 1e9:.1f} GB from {base}")
    for f in todo:
        _fetch(f"{base}/{f}", library_dir / f, remote["files"][f]["bytes"], display, f)
        if _sha256(library_dir / f) != remote["files"][f]["sha256"]:
            (library_dir / f).unlink()
            raise RuntimeError(f"{f} was corrupted in transit (checksum mismatch). Run `xai4chem setup` again.")
    for n in names:
        local["features"][n] = remote["features"][n]
    write_manifest(library_dir, local)
    return library_dir


# --- setup ------------------------------------------------------------------------------------------


def setup_library(source=None, featurizers=("datamol",), library_dir=None, force=False, seed=42, display=SILENT):
    """Download Ersilia's library (``source`` None or ``"ersilia"``) or build one from a CSV of SMILES.

    Anything already present is kept unless ``force``. Returns the library directory.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    library_dir = Path(library_dir or default_library_dir())
    path = library_dir / LIBRARY_JSON
    manifest = json.loads(path.read_text()) if path.exists() else None
    if manifest is not None and (force or manifest.get("schema") != SCHEMA):
        if not force:
            display.note("replacing a library built by an older xai4chem", "warning")
        _clear(library_dir, manifest)
        manifest = None
    ersilia = source in (None, "ersilia") and (manifest is None or manifest.get("name") == ERSILIA_LIBRARY_NAME)
    wanted = "ersilia" if source in (None, "ersilia") else str(Path(source).resolve())
    if manifest is not None and (manifest.get("source") or "ersilia") != wanted and source is not None:
        raise ValueError(
            f"{library_dir} already holds the library built from {manifest.get('source')}. Pass --force to rebuild it "
            f"from {source}, or --library-dir to keep both."
        )
    if ersilia:
        display.step(1, 2, "Downloading the Ersilia reference library")
        download_library(library_dir, featurizers, display)
        library = ReferenceLibrary(library_dir)
        display.note(f"{library.name} · {len(library):,} molecules · {', '.join(library.featurizers)}")
        display.step(2, 2, "Fingerprint cache")
        _fingerprint_cache(library_dir)
        return library_dir
    if manifest is not None:
        smiles = ReferenceLibrary(library_dir).smiles
    else:
        smiles = _read_library_smiles(source, display)
    return build_library(
        smiles,
        library_dir,
        manifest["name"] if manifest else Path(source).stem,
        featurizers,
        wanted,
        seed=seed,
        display=display,
    )


def _clear(library_dir, manifest):
    """Remove the library's own files, and nothing else a user may keep in the same folder."""
    names = {LIBRARY_JSON, SMILES_NAME, PANEL_NAME, BIT_DICTIONARY_NAME, MORGAN_BITS_INDICES, MORGAN_BITS_INDPTR}
    names |= {f"{n}.h5" for n in FEATURIZERS}
    names |= {
        Path(e["file"]).name
        for e in [manifest.get("smiles", {}), *manifest.get("features", {}).values()]
        if "file" in e
    }
    for name in names:
        (Path(library_dir) / name).unlink(missing_ok=True)


def _read_library_smiles(source, display):
    from rdkit import Chem

    from xai4chem.data.io import read_smiles

    smiles = read_smiles(source)
    valid = list(dict.fromkeys(s for s in smiles if Chem.MolFromSmiles(s) is not None))
    if len(valid) < len(smiles):
        display.note(f"dropped {len(smiles) - len(valid):,} unparseable or duplicate SMILES", "warning")
    return valid


def export_smiles(library_dir, output):
    """Write the library's SMILES, in library order, for a teacher to score."""
    import csv

    library = ReferenceLibrary(library_dir)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        writer.writerows([s] for s in library.smiles)
    return len(library)
