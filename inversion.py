from yaml import SafeDumper
import yaml
import argparse

from inversion_script import main


SafeDumper.add_representer(
    type(None),
    lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', '')
    )


def inversion_setup(yaml_file, model_name, method):
    '''
    Read the config.yaml file, and update it as needed. We took advantage
    of the already defined cluster configuration in the file.
    '''
    with open(yaml_file, 'r') as infile:
        data = yaml.full_load(infile)
        print(data.get("solver_params"))
        data['forward'] = False

        if model_name == 'marmousi2':
            data['fwi'] = True
            data['vmin'] = 1.377
            data['vmax'] = 4.688
            data['opt_meth'] = 'LBFGS'
            data['mute_depth'] = 12
        else:
            data['fwi'] = False
            data['opt_method'] = method
            data['mute_depth'] = None
        # solver parameters
        data['solver_params']['parfile_path'] = "./"+args.model+"/parameters_hdf5/"
        data['solver_params']['shotfile_path'] = "./"+args.model+"/shots/"

    with open(yaml_file, 'w') as outfile:
        yaml.safe_dump(data, outfile, default_flow_style=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model')
    parser.add_argument('method', type=str, help='gradient-based method, '
                        'valid options are LBFGS, PNLCG, PSTD)')
    args = parser.parse_args()
    inversion_setup("./config/config.yaml", args.model, args.method)
    main()
