from __future__ import annotations

import pytest

from app.services.csv_parser import CSVParseError, parse_connections_csv
from app.services.dedup import dedupe_rows
from tests.conftest import SAMPLE_CSV


def test_parse_skips_preamble_and_maps_columns():
    res = parse_connections_csv(SAMPLE_CSV)
    assert res.total_data_rows == 4
    assert len(res.usable) == 3  # Jane, John, Dup-Jane (NoUrl excluded)
    jane = res.usable[0]
    assert jane.first_name == "Jane"
    assert jane.linkedin_url == "https://www.linkedin.com/in/jane-smith"
    assert jane.company == "Google"
    assert len(res.skipped) == 1
    assert res.skipped[0]["name"] == "NoUrl Person"


def test_dedup_by_public_identifier():
    res = parse_connections_csv(SAMPLE_CSV)
    dd = dedupe_rows(res.usable)
    assert len(dd.unique) == 2
    assert len(dd.duplicates) == 1
    assert dd.duplicates[0]["public_identifier"] == "jane-smith"


def test_missing_url_column_raises():
    with pytest.raises(CSVParseError):
        parse_connections_csv(b"First Name,Last Name,Company\nA,B,C\n")


def test_empty_file_raises():
    with pytest.raises(CSVParseError):
        parse_connections_csv(b"")


def test_no_preamble_still_parses():
    csv = b"First Name,Last Name,URL\nAmy,Lee,https://www.linkedin.com/in/amy-lee\n"
    res = parse_connections_csv(csv)
    assert len(res.usable) == 1
    assert res.usable[0].linkedin_url == "https://www.linkedin.com/in/amy-lee"


def test_semicolon_delimiter():
    csv = b"First Name;Last Name;URL\nAmy;Lee;https://www.linkedin.com/in/amy-lee\n"
    res = parse_connections_csv(csv)
    assert len(res.usable) == 1


def test_upload_endpoint(client):
    r = client.post(
        "/datasets",
        files={"file": ("Connections.csv", SAMPLE_CSV, "text/csv")},
        data={"name": "My Network"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["imported"] == 2
    assert body["duplicates_removed"] == 1
    assert body["skipped_no_url"] == 1
    assert body["dataset"]["connection_count"] == 2

    ds_id = body["dataset"]["dataset_id"]
    people = client.get(f"/datasets/{ds_id}/people").json()
    assert len(people) == 2
    assert {p["full_name"] for p in people} == {"Jane Smith", "John Doe"}


def test_delete_dataset(client):
    r = client.post("/datasets", files={"file": ("c.csv", SAMPLE_CSV, "text/csv")})
    ds_id = r.json()["dataset"]["dataset_id"]
    assert client.delete(f"/datasets/{ds_id}").status_code == 204
    assert client.get(f"/datasets/{ds_id}").status_code == 404


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # llm config block, no secrets, no live calls — Anthropic is the only provider
    llm = body["llm"]
    assert isinstance(llm["priority"], list)
    assert set(llm) >= {"priority", "anthropic", "circuit_breakers"}
    assert "groq" not in llm and "openrouter" not in llm
    assert "api_key" not in str(llm) and "sk-" not in str(llm)
