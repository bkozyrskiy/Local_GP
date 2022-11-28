import os, json 

def read_config():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')) as f:
        config = json.load(f)
    return config

global config
config = read_config()

