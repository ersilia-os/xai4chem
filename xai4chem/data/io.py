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

SMILES_NAMES = ("smiles", "input", "canonical_smiles", "isomeric_smiles", "smi")
ID_NAMES = ("id", "name", "key", "compound_id", "molecule")
HARD_SEPARATORS = ("_", "-")


def read_table(path):
    """Read a CSV, TSV or parquet table, by its extension."""
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


def id_column(df, smiles_col):
    """A column naming each molecule (``id``, ``name``, ``key``…), or ``None``."""
    for c in df.columns:
        if c != smiles_col and str(c).strip().lower() in ID_NAMES:
            return c
    return None


def check_smiles_column(df, column, path, flag=None):
    """Refuse a "SMILES" column that is not SMILES (e.g. an id column picked as the first column)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    values = df[column].dropna().astype(str)
    sample = values.sample(min(len(values), 500), random_state=0) if len(values) else values
    ok = sum(Chem.MolFromSmiles(v) is not None for v in sample)
    if len(sample) and ok / len(sample) < 0.5:
        raise ValueError(
            f"Column {column!r} of {path} does not look like SMILES ({ok}/{len(sample)} parse). Name it `smiles` or "
            f"`input`{f', or pass {flag}' if flag else ''}. Columns: {list(df.columns)}"
        )


def read_smiles(path, column=None):
    """The SMILES of a table, as strings, with the column found by name or convention."""
    df = read_table(path)
    sc = smiles_column(df, column)
    check_smiles_column(df, sc, path)
    return df[sc].astype(str).tolist()


def read_labels(path, smiles_col=None, label_cols=None, flag=None):
    """A labels table → (SMILES list, DataFrame of numeric label columns).

    As in Olinda, the label columns are the numeric ones *after* the SMILES column, so a leading ``id`` or
    ``key`` column is never mistaken for a label. ``label_cols`` picks columns by name from the whole table.
    """
    import pandas as pd

    df = read_table(path)
    sc = smiles_column(df, smiles_col)
    check_smiles_column(df, sc, path, flag)
    if label_cols:
        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Label columns {missing} not found in {path}. Columns: {list(df.columns)}")
        tasks = list(label_cols)
    else:
        # `Unnamed: 0` is a pandas index written by to_csv(index=True), never a label
        after = list(df.columns).index(sc) + 1
        tasks = [
            c for c in df.columns[after:] if pd.api.types.is_numeric_dtype(df[c]) and not str(c).startswith("Unnamed:")
        ]
        skipped = [c for c in df.columns[after:] if c not in tasks and not str(c).startswith("Unnamed:")]
        if skipped:  # e.g. a stray "ND" or "<10" turns a measurement column into text
            from xai4chem.ui.log import logger

            logger.warning(f"Ignoring non-numeric column(s) of {path}: {', '.join(map(str, skipped))}")
    if not tasks:
        raise ValueError(
            f"{path} has no numeric label columns after the SMILES column {sc!r} (labels are the numeric columns "
            "to its right)."
        )
    return df[sc].astype(str).tolist(), df[tasks].reset_index(drop=True)


def match_hard_columns(soft_columns, hard_columns, strict=True):
    """Map each hard-label column onto the soft-label column it refines (Olinda's rule).

    With ``strict=False``, columns that match no soft column are left out (and reported by the caller)
    instead of raising, so an ``id`` or ``year`` column in a measurements file does not abort a fit.
    Ambiguous and doubly-claimed columns always raise.
    """
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
            if strict:
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
