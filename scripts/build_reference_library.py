"""Build the published Ersilia reference library that ``xai4chem setup`` downloads (maintainers only).

    python scripts/build_reference_library.py --smiles erl0.h5:input --out build/erl0 \
        --reuse morgan=erl0.h5:data --reuse accfg=old/accfg.h5:X --reuse datamol=old/datamol.h5:X \
        --configs old/library.json --panel old/panel.npy --bit-dictionary old/bit_dictionary.json

``--smiles`` is a CSV with a ``smiles`` column, or ``file.h5:dataset``. Every featurizer not passed with
``--reuse`` is computed from the SMILES (sequentially: about 60 min for AccFG and 100 min for datamol over
1.36M molecules). A reused matrix must hold the same molecules in the same order; a random sample of its
rows is checked against fresh featurization before it is accepted. ``--configs`` gives the featurizer
configurations of reused matrices (datamol's fitted medians), as a library manifest.

The output directory is uploaded as is (the fingerprint cache is left out; ``setup`` builds it locally):

    aws s3 sync build/erl0 s3://eosvc-public/xai4chem/erl0/ --exclude "morgan_bits_*" --exclude "*.tmp"
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from xai4chem.data.build import build_library, publish_manifest
from xai4chem.data.library import BIT_DICTIONARY_NAME, ERSILIA_LIBRARY_NAME, PANEL_NAME
from xai4chem.featurizers import FEATURIZERS, get_featurizer
from xai4chem.ui.console import ConsoleDisplay, rule

VERIFY_ROWS = 500
READ_CHUNK = 100_000


def read_source(spec):
    if ":" in spec and spec.split(":")[0].endswith(".h5"):
        path, dataset = spec.split(":", 1)
        with h5py.File(path, "r") as f:
            raw = f[dataset][:]
        return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
    from xai4chem.data.io import read_smiles

    return read_smiles(spec)


def read_matrix(spec, sparse):
    """A dense HDF5 dataset, read in chunks; as CSR for count featurizers."""
    path, dataset = spec.split(":", 1)
    with h5py.File(path, "r") as f:
        ds = f[dataset]
        blocks = [ds[i : i + READ_CHUNK] for i in range(0, ds.shape[0], READ_CHUNK)]
    if sparse:
        return sp.vstack([sp.csr_matrix(b, dtype=np.float32) for b in blocks], format="csr")
    return np.vstack(blocks).astype(np.float32)


def verify(name, X, config, smiles, seed=0):
    featurizer = FEATURIZERS[name].from_config(config)
    rows = np.sort(np.random.default_rng(seed).choice(len(smiles), size=min(VERIFY_ROWS, len(smiles)), replace=False))
    fresh = featurizer.transform_raw([smiles[i] for i in rows]).astype(np.float32)
    stored = X[rows].toarray() if sp.issparse(X) else X[rows]
    if not np.allclose(stored, fresh, rtol=1e-4, atol=1e-4, equal_nan=True):
        bad = int((~np.isclose(stored, fresh, rtol=1e-4, atol=1e-4, equal_nan=True)).any(axis=1).sum())
        raise SystemExit(f"{name}: {bad}/{len(rows)} sampled rows differ from fresh featurization; not reusing it.")
    print(f"  {name}: {len(rows)} sampled rows match fresh featurization")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default=ERSILIA_LIBRARY_NAME)
    parser.add_argument("--featurizers", default=",".join(FEATURIZERS))
    parser.add_argument("--reuse", action="append", default=[], help="name=file.h5:dataset")
    parser.add_argument("--configs", default=None, help="a library.json with the configs of reused featurizers")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--bit-dictionary", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rule(f"build the {args.name} library")
    smiles = read_source(args.smiles)
    print(f"  {len(smiles):,} molecules")
    configs = json.loads(Path(args.configs).read_text())["features"] if args.configs else {}
    reuse = {}
    for item in args.reuse:
        name, spec = item.split("=", 1)
        config = configs.get(name, {}).get("config") or get_featurizer(name).to_config()
        X = read_matrix(spec, FEATURIZERS[name]().sparse)
        if X.shape[0] != len(smiles):
            raise SystemExit(f"{name}: {X.shape[0]:,} rows for {len(smiles):,} molecules")
        verify(name, X, config, smiles)
        reuse[name] = (X, config)
    if args.panel:
        shutil.copy(args.panel, out / PANEL_NAME)
    if args.bit_dictionary:
        shutil.copy(args.bit_dictionary, out / BIT_DICTIONARY_NAME)
    build_library(
        smiles, out, args.name, args.featurizers.split(","), source="ersilia", reuse=reuse, display=ConsoleDisplay()
    )
    manifest = publish_manifest(out)
    size = sum(f["bytes"] for f in manifest["files"].values())
    print(f"  ready to upload: {len(manifest['files'])} files, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
