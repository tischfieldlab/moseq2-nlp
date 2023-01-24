import os
import pkgutil
import subprocess
from importlib import import_module
from pathlib import Path

import click
import pytest

from moseq2_nlp.cli import cli


def command_tree(obj):
    if isinstance(obj, click.Group):
        return {name: command_tree(value) for name, value in obj.commands.items()}


commands = list(command_tree(cli).keys())

@pytest.mark.parametrize("entry_point", commands, ids=commands)
def test_entry_point(entry_point):
    os.environ['COVERAGE_PROCESS_START'] = '1'
    rtn_code = subprocess.call(['moseq2-nlp', str(entry_point), '--help'])
    assert rtn_code == 0
    os.environ.pop('COVERAGE_PROCESS_START')



pkg_path = Path(__file__).resolve().parent.parent.joinpath('moseq2_nlp')
modules_to_test = pkgutil.iter_modules([pkg_path], prefix='moseq2_nlp.')
module_names = [m.name for m in modules_to_test]

@pytest.mark.parametrize("module_path", module_names)
def test_import(module_path):
    import_module(module_path)
    assert True
