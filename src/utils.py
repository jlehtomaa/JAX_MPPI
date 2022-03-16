import yaml

def read_config(cfg_file: str) -> dict:
    """ Translates a yaml config file to a dict.

    Arguments:
    ----------
    cfg_file : config file path
    
    """
    with open(cfg_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None