import yaml
import argparse

from forward_script import main as parallel
from serial_modeling import main as serial


def forward_setup(yaml_file, solver):
    '''
    Read the config.yaml file, and update it as needed. We took advantage
    of the already defined cluster configuration in the file.
    '''
    with open(yaml_file, 'r') as infile:
        data = yaml.full_load(infile)
        # solver parameters
        data['solver_params']['use_solver'] = solver

    with open(yaml_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', dest='feature', action='store_true')
    parser.add_argument('--no-serial', dest='feature', action='store_false')
    parser.add_argument('--solver', dest='solver', action='store_true')
    parser.add_argument('--no-solver', dest='solver', action='store_false')
    parser.set_defaults(feature=True)
    parser.set_defaults(solver=True)
    args = parser.parse_args()
    forward_setup('./config/config.yaml', args.solver)
    if args.feature:
        serial()
    else:
    	parallel()
