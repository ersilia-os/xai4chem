# Where the reference library comes from, and sharing it (eosvc)

Everything in a library directory is model-independent. Ersilia's library (`erl0`, 1,355,109 molecules)
is published ready-made in the eosvc public bucket, and `xai4chem setup` downloads it:

```
s3://eosvc-public/xai4chem/data/erl0/
  library.json          manifest, with the size and SHA-256 of every file
  smiles.h5             the SMILES, in row order                           0.1 GB
  morgan.h5             Morgan counts (identical to Olinda's), CSR          0.3 GB
  accfg.h5              AccFG counts, CSR                                   0.06 GB
  datamol.h5            datamol descriptors, 1.36M × 22 float32             0.1 GB
  panel.npy             the explanation panel (row indices)
  bit_dictionary.json   Morgan bit names and collision rates (from 100k molecules)
```

`setup` fetches only the featurizers asked for (Morgan and datamol by default), checks every file against
its checksum, and builds the binary-fingerprint cache (`morgan_bits_*.npy`) locally. Set
`XAI4CHEM_LIBRARY_URL` to download from a mirror or a local folder instead.

## Publishing a new version (maintainers)

```bash
python scripts/build_reference_library.py --smiles erl0_smiles.csv --out data/erl0
aws s3 sync data/erl0 s3://eosvc-public/xai4chem/data/erl0/ --exclude "morgan_bits_*" --exclude "*.tmp"
```

The script featurizes the SMILES (about 2 hours, sequentially) or reuses matrices computed earlier, after
checking a random sample of their rows against fresh featurization (see its docstring).

## Sharing a library directory

A library directory is relocatable: `library.json` records its files relative to the directory, so
copying it anywhere, including through [eosvc](https://github.com/ersilia-os/eosvc), gives a working
library. This is how to share a library built from your own SMILES:

```bash
xai4chem setup --library my_smiles.csv -f all --library-dir data/library
eosvc upload --path data/library                     # needs the repository's access.json
```

and elsewhere:

```bash
eosvc download --path data/library
export XAI4CHEM_HOME=$PWD/data     # the library is then $XAI4CHEM_HOME/library
```

Models are directories too (boosters, metadata, explanations, report), typically 30–70 MB each. They can
be shared the same way. A model finds its library in this order:
1. `--library-dir`
2. the directory it was trained with
3. `~/.xai4chem/library`

It accepts the first whose name and size match.
