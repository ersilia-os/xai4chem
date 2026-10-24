"""Reading input tables: SMILES files, soft labels and hard labels.

Follows Olinda's conventions, so the same files work for both tools:

- CSV, TSV (``.tsv``/``.tab``) or Parquet, chosen by extension.
- The SMILES column is ``smiles`` or ``input`` (case-insensitive), else the first column. Ersilia and
  Isaura outputs (``key, input, ...``) therefore work as they are.
- Label columns are every *numeric* column other than the SMILES. A ``key`` column, or any other text
  column, is ignored.
- A hard-label column matches a soft-label column by exact name, or by being a prefix of exactly one of
  them up to ``_`` or ``-`` (``abaumannii_inhibition`` → ``abaumannii_inhibition_probability``).
"""

from pathlib import Path

SMILES_NAMES = ("smiles", "input")
HARD_SEPARATORS = ("_", "-")


def read_table(path):
    import pandas as pd

    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def smiles_column(df, name=None):
    """Name of the SMILES column of ``df`` (see module docstring)."""
    if name is not None:
        if name not in df.columns:
            raise ValueError(f"SMILES column {name!r} not found. Columns: {list(df.columns)}")
        return name
    for c in df.columns:
        if str(c).strip().lower() in SMILES_NAMES:
            return c
    return df.columns[0]


def read_smiles(path, column=None):
    df = read_table(path)
    return df[smiles_column(df, column)].astype(str).tolist()


def read_labels(path, smiles_col=None, label_cols=None):
    """A labels table → (SMILES list, DataFrame of numeric label columns)."""
    import pandas as pd

    df = read_table(path)
    sc = smiles_column(df, smiles_col)
    if label_cols:
        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Label columns {missing} not found in {path}. Columns: {list(df.columns)}")
        tasks = list(label_cols)
    else:
        tasks = [c for c in df.columns if c != sc and pd.api.types.is_numeric_dtype(df[c])]
    if not tasks:
        raise ValueError(f"{path} has no numeric label columns besides the SMILES column {sc!r}.")
    return df[sc].astype(str).tolist(), df[tasks].reset_index(drop=True)


def match_hard_columns(soft_columns, hard_columns):
    """Map each hard-label column onto the soft-label column it refines (Olinda's rule)."""
    soft = [str(c) for c in soft_columns]
    mapping, problems = {}, []
    for raw in hard_columns:
        hard = str(raw)
        if hard in soft:
            mapping[hard] = hard
            continue
        candidates = [s for s in soft if any(s.startswith(hard + sep) for sep in HARD_SEPARATORS)]
        if len(candidates) == 1:
            mapping[hard] = candidates[0]
        elif not candidates:
            problems.append(f"hard column '{hard}' matches no soft column")
        else:
            problems.append(f"hard column '{hard}' is ambiguous: matches {candidates}")
    claimed = {}
    for hard, target in mapping.items():
        if target in claimed:
            problems.append(f"soft column '{target}' claimed by both '{claimed[target]}' and '{hard}'")
        claimed[target] = hard
    if problems:
        raise ValueError(
            "could not align hard labels with soft labels:\n  - " + "\n  - ".join(problems) + f"\nsoft columns: {soft}"
        )
    return mapping
