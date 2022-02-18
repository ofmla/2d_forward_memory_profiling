import os
import errno
import yaml
import argparse

from forward_script import main


def model_to_dict(argument):
    '''
    Get forward modeling configurations of the model. Model name
    is looked up against the switcher dictionary mapping.
    '''
    switcher = {
        'marmousi2': {'src_depth': 40.0, 'rec_depth': 80.0,
                      'nrecs': 426, 'nshots': 16,
                      'solver_params': {
                          'shotfile_path': './',
                          'parfile_path': './marmousi2/parameters_hdf5/',
                          'setup_func': 'tti', 't0': 0.0, 'tn': 5000.0,
                          'dt': 4.0, 'f0': 0.004, 'model_name': 'marmousi2',
                          'nbl': 50, 'space_order': 8, 'born': False,
                          'dtype': 'float32'}},
        'marmousi': {'src_depth': 12.5, 'rec_depth': 12.5,
                     'nrecs': 369, 'nshots': 123,
                     'solver_params': {
                         'shotfile_path': './',
                         'parfile_path': './marmousi/parameters_hdf5/',
                         'setup_func': 'tti', 't0': 0.0, 'tn': 2500.0,
                         'dt': 4.0, 'f0': 0.008, 'model_name': 'marmousi',
                         'nbl': 50, 'space_order': 8, 'born': True,
                         'dtype': 'float32'}},
        'overthrust': {'src_depth': 10.0, 'rec_depth': 10.0,
                       'nrecs': 301, 'nshots': 151,
                       'solver_params': {
                           'shotfile_path': './',
                           'parfile_path': './overthrust/parameters_hdf5/',
                           'setup_func': 'tti', 't0': 0.0, 'tn': 2500.0,
                           'dt': 4.0, 'f0': 0.02, 'model_name': 'overthrust',
                           'nbl': 50, 'space_order': 8, 'born': True,
                           'dtype': 'float32'}},
    }

    return switcher.get(argument)


def forward_setup(yaml_file, model_name):
    '''
    Read the config.yaml file, and update it as needed. We took advantage
    of the already defined cluster configuration in the file.
    '''
    with open(yaml_file, 'r') as infile:
        data = yaml.full_load(infile)
        print(data.get("solver_params"))
        data['forward'] = True
        cfg = model_to_dict(model_name)
        # geometry
        data['src_depth'] = cfg['src_depth']
        data['rec_depth'] = cfg['rec_depth']
        data['nrecs'] = cfg['nrecs']
        data['nshots'] = cfg['nshots']
        # solver parameters
        data['solver_params'] = cfg['solver_params']
        data['solver_params']['shotfile_path'] = "./"+args.model+"/shots/"

    with open(yaml_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=None)


def make_sure_path_exists(path):
    '''Create a folder within a given path.'''
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model')
    args = parser.parse_args()
    l = ['marmousi', 'marmousi2', 'overthrust']
    if args.model not in l:
        msg_strng = "{} model is not found; model must be one of the following: {}"
        raise ValueError(msg_strng.format(args.model, ", ".join(l)))
    make_sure_path_exists("./"+args.model+"/shots")
    forward_setup("./config/config.yaml", args.model)
    main()
