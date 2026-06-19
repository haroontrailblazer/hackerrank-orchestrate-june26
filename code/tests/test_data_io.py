"""CSV + path I/O against the real dataset files and round-trips."""

import csv

from argus.config import Settings
from argus.constants import OUTPUT_COLUMNS
from argus import data_io


def test_split_images_strips_and_drops_empty():
    assert data_io._split_images("a.jpg ; b.jpg;") == ["a.jpg", "b.jpg"]


def test_load_evidence_rules_expands_all_to_each_object():
    lookup = data_io.load_evidence_rules(Settings().evidence_csv)
    # 'all' / 'general claim review' must be reachable per concrete object
    assert ("car", "general claim review") in lookup
    assert ("laptop", "general claim review") in lookup
    # an object-specific family
    assert ("car", "dent or scratch") in lookup


def test_load_user_history_parses_ints_and_flags():
    hist = data_io.load_user_history(Settings().user_history_csv)
    assert "user_001" in hist
    assert isinstance(hist["user_001"].past_claim_count, int)
    assert "user_history_risk" in hist["user_005"].history_flags


def test_write_output_roundtrip_exact_columns_and_quoting(tmp_path):
    row = {c: f"v_{c}" for c in OUTPUT_COLUMNS}
    out = tmp_path / "out.csv"
    data_io.write_output(out, [row])

    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        first = next(reader)
    assert header == OUTPUT_COLUMNS  # exact order
    assert first == [f"v_{c}" for c in OUTPUT_COLUMNS]
    # QUOTE_ALL: every field quoted
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith('"user_id"')
