import json
from pathlib import Path

import pytest

from thimiko.cli import main, parse_args


def test_bare_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "Build, update, and search local chat history." in output
    assert "{build,update,search,list,mcp}" in output


def test_mcp_remains_explicit() -> None:
    assert parse_args(["mcp"]).command == "mcp"


def test_search_accepts_gemini_source() -> None:
    assert parse_args(["search", "migration", "--source", "gemini"]).source == "gemini"


def test_list_verbose_flag() -> None:
    assert parse_args(["list", "-v"]).verbose is True
    assert parse_args(["list"]).verbose is False


def test_list_names_every_registered_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0

    payload = json.loads(capsys.readouterr().out)
    names = [source["name"] for source in payload["sources"]]
    assert names == ["codex", "claude", "copilot", "gemini"]
    assert "indexed" not in payload["sources"][0]


def test_list_verbose_without_index_reports_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "missing" / "thimiko.sqlite"

    assert main(["--db", str(db_path), "list", "-v"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed"] is False
    assert all(source["indexed"] == 0 for source in payload["sources"])
    assert all("on_disk" in source for source in payload["sources"])
    assert not db_path.exists()
