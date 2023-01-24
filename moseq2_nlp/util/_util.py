import errno
import os
from gettext import gettext
from typing import Any, Dict, List, Optional, Sequence, Type

import click
import ruamel.yaml as yaml
from click.shell_completion import CompletionItem


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name: str) -> Type[click.Command]:
    """Create and return a class inheriting `click.Command` which accepts a configuration file
        containing arguments/options accepted by the command.

        The returned class should be passed to the `@click.Commnad` parameter `cls`:

        ```
        @cli.command(name='command-name', cls=command_with_config('config_file'))
        ```

    Parameters:
        config_file_param_name (str): name of the parameter that accepts a configuration file

    Returns:
        class (Type[click.Command]): Class to use when constructing a new click.Command
    """

    class custom_command_class(click.Command):
        def invoke(self, ctx):
            # grab the config file
            config_file = ctx.params[config_file_param_name]
            param_defaults = {p.human_readable_name: p.default for p in self.params if isinstance(p, click.core.Option)}
            param_defaults = {k: tuple(v) if type(v) is list else v for k, v in param_defaults.items()}
            param_cli = {k: tuple(v) if type(v) is list else v for k, v in ctx.params.items()}

            if config_file is not None:

                config_data = read_yaml(config_file)
                # modified to only use keys that are actually defined in options
                config_data = {
                    k: tuple(v) if isinstance(v, yaml.comments.CommentedSeq) else v
                    for k, v in config_data.items()
                    if k in param_defaults.keys()
                }

                # find differences btw config and param defaults
                diffs = set(param_defaults.items()) ^ set(param_cli.items())

                # combine defaults w/ config data
                combined = {**param_defaults, **config_data}

                # update cli params that are non-default
                keys = [d[0] for d in diffs]
                for k in set(keys):
                    combined[k] = ctx.params[k]

                ctx.params = combined

            return super().invoke(ctx)

    return custom_command_class


def get_command_defaults(command: click.Command):
    """Get the defualt values for the options of `command`"""
    return {tmp.name: tmp.default for tmp in command.params if not tmp.required}


class IntChoice(click.ParamType):
    name = "intchoice"

    def __init__(self, choices: Sequence[int]) -> None:
        self.choices = choices

    def to_info_dict(self) -> Dict[str, Any]:
        info_dict = super().to_info_dict()
        info_dict["choices"] = self.choices
        return info_dict

    def get_metavar(self, param: click.Parameter) -> str:
        choices_str = "|".join([str(c) for c in self.choices])

        # Use curly braces to indicate a required argument.
        if param.required and param.param_type_name == "argument":
            return f"{{{choices_str}}}"

        # Use square braces to indicate an option or optional argument.
        return f"[{choices_str}]"

    def get_missing_message(self, param: click.Parameter) -> str:
        return gettext("Choose from:\n\t{choices}").format(choices=",\n\t".join([str(c) for c in self.choices]))

    def convert(self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> Any:
        return int(str(value))

    def __repr__(self) -> str:
        return f"Choice({list([str(c) for c in self.choices])})"

    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> List[CompletionItem]:
        """Complete choices that start with the incomplete value.

        :param ctx: Invocation context for this command.
        :param param: The parameter that is requesting completion.
        :param incomplete: Value being completed. May be empty.

        .. versionadded:: 8.0
        """

        str_choices = map(str, self.choices)

        incomplete = incomplete.lower()
        matched = (c for c in str_choices if c.lower().startswith(incomplete))

        return [CompletionItem(c) for c in matched]


def read_yaml(yaml_file: str) -> dict:
    """Read a yaml file into dict object

    Parameters:
        yaml_file (str): path to yaml file

    Returns:
        return_dict (dict): dict of yaml contents
    """
    with open(yaml_file, "r") as f:
        yml = yaml.YAML(typ="safe")
        return yml.load(f)


def write_yaml(yaml_file: str, data: dict) -> None:
    """Write a dict object into a yaml file

    Parameters:
        yaml_file (str): path to yaml file
        data (dict): dict of data to write to `yaml_file`
    """
    with open(yaml_file, "w") as f:
        yml = yaml.YAML(typ="safe")
        yml.default_flow_style = False
        yml.dump(data, f)


def ensure_dir(path: str) -> str:
    """Creates the directories specified by path if they do not already exist.

    Parameters:
        path (str): path to directory that should be created

    Returns:
        return_path (str): path to the directory that now exists
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            # if the exception is raised because the directory already exits,
            # than our work is done and everything is OK, otherwise re-raise the error
            # THIS CAN OCCUR FROM A POSSIBLE RACE CONDITION!!!
            if exception.errno != errno.EEXIST:
                raise
    return path


def get_unique_list_elements(lst):
    """Returns unique elements from list

    Args:
        lst: the list

    Returns:
        unique_elements: the unique elements
    """
    unique_elements = []
    for el in lst:
        if el not in unique_elements:
            unique_elements.append(el)
    return unique_elements
