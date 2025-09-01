"""
Tests for SciDoc CLI.

This module contains tests for the command-line interface.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from scidoc.cli import app


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


def test_cli_help(runner):
    """Test that CLI help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "SciDoc" in result.output


def test_cli_explore_help(runner):
    """Test that explore command help works."""
    result = runner.invoke(app, ["explore", "--help"])
    assert result.exit_code == 0
    assert "explore" in result.output


def test_cli_init_help(runner):
    """Test that init command help works."""
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    assert "init" in result.output
