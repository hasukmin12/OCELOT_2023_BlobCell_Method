import sys
import importlib
sys.path.append('../')
def call_trans_function(config):
    mod = importlib.import_module(f'transforms.trans_v{config["TRANSFORM"]}')
    train_transforms, val_transforms = mod.call_transforms(config)
    return train_transforms, val_transforms


def call_trans_function_for_semi(config):
    mod = importlib.import_module(f'transforms.trans_v{config["TRANSFORM"]}')
    train_transforms, val_transforms, unlabel_transforms = mod.call_transforms_for_semi(config)
    return train_transforms, val_transforms, unlabel_transforms


def call_trans_function_for_semi_easy_hard(config):
    mod = importlib.import_module(f'transforms.trans_v{config["TRANSFORM"]}')
    hard_train_transforms, val_transforms, easy_unlabel_transforms, hard_unlabel_transforms = mod.call_transforms_for_semi_easy_hard(config)
    return hard_train_transforms, val_transforms, easy_unlabel_transforms, hard_unlabel_transforms      