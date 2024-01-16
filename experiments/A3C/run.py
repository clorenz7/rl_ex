import argparse
import datetime
import os

import jstyleson

from cor_rl.utils import DEFAULT_DIR
from cor_rl import a3c


class ExperimentParams(dict):

    DEFAULT_AGENT = {}
    DEFAULT_TRAIN = {}
    DEFAULT_ENV = {}
    DEFAULT_SIM = {
        'experiment_name': datetime.datetime.now().strftime('%Y_%b_%d_%H_%M')
    }

    def __init__(self, agent_params={}, train_params={},
                 env_params={}, simulation_params={}):
        self['agent_params'] = dict(self.DEFAULT_AGENT)
        self['agent_params'].update(agent_params)
        self['train_params'] = dict(self.DEFAULT_TRAIN)
        self['train_params'].update(train_params)
        self['env_params'] = dict(self.DEFAULT_ENV)
        self['env_params'].update(env_params)
        self['simulation_params'] = dict(self.DEFAULT_SIM)
        self['simulation_params'].update(simulation_params)


def main():

    parser = argparse.ArgumentParser(
        description="Train an A3C Agent"
    )
    parser.add_argument(
        '-j', "--json", type=str, default="",
        help="Json experiment parameters"
    )
    parser.add_argument(
        '-o', '--out_dir', default=DEFAULT_DIR,
        help="Where to store results"
    )

    # Get command line parameters and setup output directory
    cli_args = parser.parse_args()
    os.makedirs(cli_args.out_dir, exist_ok=True)

    # Parse and prepare experiment parameters
    if cli_args.json:
        with open(cli_args.json, 'r') as fp:
            json_params = jstyleson.load(fp)

        base_name = os.path.basename(cli_args.json).rsplit(".", 1)[0]
        sim_params = json_params.get('simulation_params', {})
        if sim_params.get('run_name') is None:
            sim_params['run_name'] = base_name
            json_params['simmulatiion_params'] = sim_params

    else:
        json_params = {}

    experiment_params = ExperimentParams(**json_params)

    a3c.train_loop_continuous(
        out_dir=cli_args.out_dir,
        **experiment_params
    )


if __name__ == '__main__':
    main()
