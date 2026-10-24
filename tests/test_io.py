import pandas as pd
import pytest

from xai4chem.io import match_hard_columns, read_labels, smiles_column


def test_smiles_column_conventions():
    assert smiles_column(pd.DataFrame(columns=["key", "input", "y"])) == "input"
    assert smiles_column(pd.DataFrame(columns=["ID", "SMILES", "y"])) == "SMILES"
    assert smiles_column(pd.DataFrame(columns=["mol", "y"])) == "mol"


def test_read_labels_skips_text_columns(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"key": ["a", "b"], "input": ["CCO", "CCC"], "p": [0.1, 0.9], "note": ["x", "y"]}).to_csv(
        p, index=False
    )
    smiles, Y = read_labels(p)
    assert smiles == ["CCO", "CCC"] and list(Y.columns) == ["p"]


def test_hard_columns_match_by_name_or_prefix():
    soft = ["abaumannii_inhibition_probability", "tox"]
    assert match_hard_columns(soft, ["abaumannii_inhibition"]) == {
        "abaumannii_inhibition": "abaumannii_inhibition_probability"
    }
    assert match_hard_columns(soft, ["tox"]) == {"tox": "tox"}
    with pytest.raises(ValueError):
        match_hard_columns(soft, ["solubility"])
    with pytest.raises(ValueError):
        match_hard_columns(["a_x", "a_y"], ["a"])
