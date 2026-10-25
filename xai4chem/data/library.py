"""Read access to a reference library: its manifest, matrices, panel and Morgan bit names.

A library directory (default ``~/.xai4chem/library/``) holds::

    library.json            where the SMILES and every feature matrix live (portable paths)
    <featurizer>.h5         dataset ``X`` (rows × features, raw dtype)
    panel.npy               row indices of a fixed random subset, used for global explanations
    bit_dictionary.json     per Morgan bit, the substructures that set it across the library
    morgan_bits_*.npy       binary Morgan fingerprints as CSR index arrays (nearest-neighbour search)

By default it is Ersilia's reference library, the 1,355,109 molecules Olinda distills over. Its Morgan
counts are Olinda's ``erl0_qmfp_r3.h5``, recorded by name and used in place. Building a library is
:mod:`xai4chem.data.build`'s job; this module only reads.
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np

from xai4chem.featurizers import featurizer_from_config
from xai4chem.featurizers.morgan import NBITS

LIBRARY_JSON = "library.json"
PANEL_NAME = "panel.npy"
BIT_DICTIONARY_NAME = "bit_dictionary.json"
MORGAN_BITS_INDICES = "morgan_bits_indices.npy"
MORGAN_BITS_INDPTR = "morgan_bits_indptr.npy"
READ_CHUNK = 100_000

ERSILIA_LIBRARY_NAME = "erl0"
ERSILIA_LIBRARY_FILENAME = "erl0_qmfp_r3.h5"
OLINDA_HOME = Path.home() / ".olinda"


def home_dir():
    return Path(os.environ.get("XAI4CHEM_HOME", Path.home() / ".xai4chem"))


def default_library_dir():
    return home_dir() / "library"


def find_ersilia_library():
    """Olinda's ``erl0_qmfp_r3.h5`` in ``~/.olinda`` or ``$XAI4CHEM_HOME``, or ``None``."""
    for candidate in (OLINDA_HOME / ERSILIA_LIBRARY_FILENAME, home_dir() / ERSILIA_LIBRARY_FILENAME):
        if candidate.exists():
            return candidate
    return None


def portable(path, library_dir):
    """How a file is recorded in ``library.json``, so that the directory can be moved or shared.

    Files inside the directory are stored relative to it; Olinda's file is stored by name and found again
    on whichever machine reads the library.
    """
    path, library_dir = Path(path), Path(library_dir)
    try:
        return str(path.resolve().relative_to(library_dir.resolve()))
    except ValueError:
        return path.name if path.name == ERSILIA_LIBRARY_FILENAME else str(path)


def resolve_file(entry, library_dir):
    """The absolute path of a ``library.json`` file entry: relative, Olinda's file by name, or absolute."""
    path = Path(entry)
    if path.is_absolute() and path.exists():
        return path
    for candidate in (Path(library_dir) / path, path, Path(library_dir) / path.name):
        # relative to the library; relative to the working directory (libraries written before 2.0);
        # or an absolute path recorded before the directory was moved
        if candidate.exists():
            return candidate.resolve()
    if path.name == ERSILIA_LIBRARY_FILENAME and find_ersilia_library() is not None:
        return find_ersilia_library()
    raise FileNotFoundError(f"{entry} (listed in {Path(library_dir) / LIBRARY_JSON}) was not found.")


def write_manifest(library_dir, manifest):
    (Path(library_dir) / LIBRARY_JSON).write_text(json.dumps(manifest, indent=2))


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
        label = f"bit {bit}: {top['smiles']} ({top['share']:.0%})"
        if top["share"] < 0.5 and len(frags) > 1:
            # A collided bit: one substructure would mislead, so name the runner-up and count the rest
            label += f" · {frags[1]['smiles']} ({frags[1]['share']:.0%})"
            rest = self.bits[str(bit)]["distinct"] - 2
            if rest > 0:
                label += f" · +{rest} more"
        return label


class ReferenceLibrary(object):
    """Read access to a library directory written by :func:`xai4chem.data.build.setup_library`."""

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
        """The library SMILES, in row order (loaded on first access; ~2 s for the Ersilia library)."""
        if self._smiles is None:
            src = self.manifest["smiles"]
            with h5py.File(resolve_file(src["file"], self.dir), "r") as f:
                raw = f[src["dataset"]][:]
            self._smiles = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
        return self._smiles

    @property
    def index(self):
        """SMILES → row."""
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
                f"The library at {self.dir} has no {name} features. Run `xai4chem setup -f {name}`."
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
        with h5py.File(resolve_file(src["file"], self.dir), "r") as f:
            ds = f[src["dataset"]]
            if rows is None:
                return ds[:]
            rows = np.asarray(rows)
            if len(rows) > 0.05 * ds.shape[0]:
                # Point selection is slow for large row sets; contiguous chunk reads are not
                return _gather(ds, rows)
            # h5py point selection needs strictly increasing indices: read each row once, then expand
            unique, inverse = np.unique(rows, return_inverse=True)
            return ds[unique][inverse]

    def csr(self, name, rows=None, tick=None):
        """Rows as float32 CSR, read in chunks so a dense copy of the library never exists.

        For count featurizers (Morgan, AccFG), whose zeros mean "absent": XGBoost treats the implicit
        zeros of a CSR matrix as missing, the same split for a count feature.
        """
        import scipy.sparse as sp

        src = self._source(name)
        with h5py.File(resolve_file(src["file"], self.dir), "r") as f:
            ds = f[src["dataset"]]
            n, width = ds.shape
            rows = np.arange(n) if rows is None else np.asarray(rows)
            order = np.argsort(rows, kind="stable")
            sorted_rows = rows[order]
            blocks, pos = [], 0
            for start in range(0, n, READ_CHUNK):
                stop = min(start + READ_CHUNK, n)
                end = np.searchsorted(sorted_rows, stop, side="left")
                if end > pos:
                    blocks.append(sp.csr_matrix(ds[start:stop][sorted_rows[pos:end] - start], dtype=np.float32))
                    pos = end
                    if tick is not None:
                        tick(pos)
        X = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, width), dtype=np.float32)
        if np.all(order[:-1] < order[1:]):
            return X
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return X[inverse]

    def morgan_bits(self):
        """Binary Morgan fingerprints of the whole library as float32 CSR (cached by ``setup``)."""
        import scipy.sparse as sp

        from xai4chem.data.similarity import binarize

        indices_path, indptr_path = self.dir / MORGAN_BITS_INDICES, self.dir / MORGAN_BITS_INDPTR
        if indices_path.exists() and indptr_path.exists():
            indices, indptr = np.load(indices_path), np.load(indptr_path)
            return sp.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(indptr) - 1, NBITS))
        X = binarize(self.csr("morgan"))
        try:
            np.save(indices_path, X.indices.astype(np.int16))
            np.save(indptr_path, X.indptr.astype(np.int64))
        except OSError:
            pass  # a read-only library still works, just without the cache
        return X

    def panel(self):
        """Row indices of the explanation panel."""
        return np.load(self.dir / PANEL_NAME)


def _gather(ds, rows):
    out = np.empty((len(rows), ds.shape[1]), dtype=ds.dtype)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    pos = 0
    for start in range(0, ds.shape[0], READ_CHUNK):
        stop = min(start + READ_CHUNK, ds.shape[0])
        end = np.searchsorted(sorted_rows, stop, side="left")
        if end > pos:
            out[order[pos:end]] = ds[start:stop][sorted_rows[pos:end] - start]
            pos = end
    return out


def library_for_model(manifest, library_dir=None):
    """The reference library a model was trained against.

    An explicit ``library_dir`` must match the model (name and size). Otherwise the directory recorded at
    training and then the default location are tried, so a library copied to another machine (for example
    through eosvc) is found without retraining.
    """
    record = manifest["library"]

    def matches(library):
        return library.name == record["name"] and len(library) == int(record["n_molecules"])

    if library_dir is not None:
        library = ReferenceLibrary(library_dir)
        if not matches(library):
            raise ValueError(
                f"{library_dir} holds library {library.name} ({len(library):,} molecules), but the model was trained on "
                f"{record['name']} ({int(record['n_molecules']):,} molecules)."
            )
        return library
    tried = []
    for candidate in (record.get("dir"), default_library_dir()):
        if candidate is None or str(candidate) in tried:
            continue
        tried.append(str(candidate))
        if (Path(candidate) / LIBRARY_JSON).exists():
            library = ReferenceLibrary(candidate)
            if matches(library):
                return library
    raise FileNotFoundError(
        f"The reference library this model was trained on ({record['name']}, {int(record['n_molecules']):,} "
        f"molecules) was not found in {', '.join(tried)}. Pass --library-dir, or run `xai4chem setup`."
    )
