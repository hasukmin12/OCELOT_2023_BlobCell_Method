import importlib

def call_config(trainer):
    mod = importlib.import_module(f'config.{trainer}')
    return mod.info, mod.config