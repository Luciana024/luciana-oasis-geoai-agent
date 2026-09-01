from pathlib import Path

import pandas as pd

from presentation.website_export import round_article_table, write_article_tables


def test_round_article_table_matches_edinburgh_digits(tmp_path: Path):
    frame = pd.DataFrame([{"MAE": 44.72907168, "MAE_skill": 0.06754353, "label": "keep"}])
    rounded = round_article_table(frame, {"MAE": 2, "MAE_skill": 3})
    assert rounded.loc[0, "MAE"] == 44.73
    assert rounded.loc[0, "MAE_skill"] == 0.068
    assert rounded.loc[0, "label"] == "keep"


def test_write_article_tables_writes_display_and_full(tmp_path: Path):
    tables = {"table02_overall_performance": pd.DataFrame([{"MAE": 44.729, "MAE_skill": 0.0675}])}
    article = write_article_tables(tmp_path / "article", tables)
    display = pd.read_csv(article / "table02_overall_performance.csv")
    full = pd.read_csv(article / "full_precision" / "table02_overall_performance.csv")
    assert display.loc[0, "MAE"] == 44.73
    assert abs(full.loc[0, "MAE"] - 44.729) < 1e-9
