from moseq2_nlp.utils import command_with_config, write_yaml
import click
import moseq2_nlp.train as trainer
import os

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass


@cli.command(name='train', cls=command_with_config('config_file'), help='train a classifier')
@click.option('--name', type=str)
@click.option('--save-dir', type=str, default=os.getcwd())
@click.option('--model-path', type=click.Path(exists=True))
@click.option('--index-path', type=click.Path(exists=True))
@click.option('--timepoint', type=int)
@click.option('--representation', type=click.Choice(['embeddings', 'usages', 'transitions']), default='embeddings')
@click.option('--emissions', is_flag=True)
@click.option('--custom-groupings', multiple=True, nargs='?', default=[])
@click.option('--num-syllables', type=int, default=70)
@click.option('--num-transitions', type=int, default=300)
@click.option('--min-count', type=int, default=1)
@click.option('--dm', type=int, default=2)
@click.option('--embedding-dim', type=int, default=300)
@click.option('--embedding-window', type=int, default=20)
@click.option('--embedding-epochs', type=int, default=50)
@click.option('--bad-syllables', multiple=True, default=[-5])
@click.option('--scoring', type=str, default='accuracy')
@click.option('--k', type=int, default=1)
@click.option('--penalty', type=str, default='l2')
@click.option('--num-c', type=int, default=11)
@click.option('--seed', type=int, default=0)
@click.option('--config-file', type=click.Path())
def train(name, save_dir, model_path, index_path, timepoint, representation, emissions, custom_groupings, num_syllables, num_transitions, min_count, dm, embedding_dim, embedding_window,
          embedding_epochs, bad_syllables, scoring, k, penalty, num_c, seed, config_file):

    trainer.train(name, save_dir, model_path, index_path, timepoint, representation, emissions, custom_groupings, num_syllables, num_transitions, min_count, dm, embedding_dim, embedding_window,
          embedding_epochs, bad_syllables, scoring, k, penalty, num_c, seed)



@cli.command(name="generate-config", help="Generates a configuration file that holds editable options for training parameters.")
@click.option('--output-file', '-o', type=click.Path(), default='config.yaml')
def generate_config(output_file):

    output_file = os.path.abspath(output_file)
    objs = train.params
    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}

    write_yaml(output_file, params)
    print(f'Successfully generated config file at "{output_file}".')


if __name__ == '__main__':
    cli()
