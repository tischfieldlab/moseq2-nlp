import os
import pkgutil
import subprocess
from importlib import import_module
from pathlib import Path
from typing import List

import click
import pytest

from moseq2_nlp.cli import cli


def command_tree(obj):
    if isinstance(obj, click.Group):
        return {name: command_tree(value) for name, value in obj.commands.items()}


def collect_commands(group: click.Group) -> List[str]:
    return list(command_tree(group).keys())


def collect_modules(root: str) -> List[str]:
    pkg_path = str(Path(__file__).resolve().parent.parent.joinpath(root))
    modules_to_test = pkgutil.iter_modules([pkg_path], prefix=f"{root}.")
    module_names = [m.name for m in modules_to_test]
    return module_names


@pytest.mark.parametrize("entry_point", collect_commands(cli))
def test_entry_point(entry_point):
    os.environ["COVERAGE_PROCESS_START"] = "1"
    rtn_code = subprocess.call(["moseq2-nlp", str(entry_point), "--help"])
    assert rtn_code == 0
    os.environ.pop("COVERAGE_PROCESS_START")


@pytest.mark.parametrize("module_path", collect_modules("paws_tools"))
def test_import(module_path):
    import_module(module_path)
    assert True
