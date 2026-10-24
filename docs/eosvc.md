# Sharing the precomputed library (eosvc)

Everything `xai4chem setup` computes is model-independent, so it only needs computing once per
organisation. A library directory is relocatable:
- `library.json` records its files relative to the directory
- Olinda's Morgan file is recorded by name, and found in the directory itself or in `~/.olinda`

Copying the directory anywhere, including through [eosvc](https://github.com/ersilia-os/eosvc), gives a
working library.

## What to store

```
data/library/
  library.json                  manifest (relative paths)
  accfg.h5                      AccFG counts, 1.36M × 536 uint8          0.7 GB
  datamol.h5                    datamol descriptors, 1.36M × 22 float32   0.1 GB
  panel.npy                     the explanation panel (row indices)
  bit_dictionary.json           Morgan bit names and collision rates
  morgan_bits_indices.npy       binary Morgan fingerprints as CSR index arrays (applicability search)
  morgan_bits_indptr.npy
  erl0_qmfp_r3.h5               optional (2.9 GB): Olinda's Morgan counts; without it, ~/.olinda is used
```

Models are directories too (boosters, metadata, explanations, report), typically 30–70 MB each. They can
be shared the same way. A model finds its library in this order:
1. `--library-dir`
2. the directory it was trained with
3. `~/.xai4chem/library`

It accepts the first whose name and size match.

## How

Build once:

```bash
xai4chem setup -f all --library-dir data/library     # ~30 min on 8 cores for the Ersilia library
eosvc upload --path data/library                     # needs the repository's access.json
```

Use anywhere:

```bash
eosvc download --path data/library
export XAI4CHEM_HOME=$PWD/data     # the library is then $XAI4CHEM_HOME/library
xai4chem setup                     # checks what is there; computes nothing if it is complete
```

Alternatively, pass `--library-dir data/library` to each command, or copy the folder to
`~/.xai4chem/library`.
