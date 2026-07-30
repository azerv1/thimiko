import pytest

from thimiko.cli import main, parse_args


def test_bare_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "Build, update, and search local chat history." in output
    assert "{build,update,search,mcp}" in output


def test_mcp_remains_explicit() -> None:
    assert parse_args(["mcp"]).command == "mcp"


def test_search_accepts_gemini_source() -> None:
    assert parse_args(["search", "migration", "--source", "gemini"]).source == "gemini"
