import pandas as pd

def test_missing_and_zero_are_distinct():
    df = pd.DataFrame({
        "Wound_Size_Length_cm": ["N/A", 0.0, 2.5],
        "Wound_Size_Width_cm": [None, 0.0, 1.5],
    })

    for col in ["Wound_Size_Length_cm", "Wound_Size_Width_cm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    assert pd.isna(df.loc[0, "Wound_Size_Length_cm"])
    assert pd.isna(df.loc[0, "Wound_Size_Width_cm"])

    assert df.loc[1, "Wound_Size_Length_cm"] == 0.0
    assert df.loc[1, "Wound_Size_Width_cm"] == 0.0