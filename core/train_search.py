import os, tempfile
import random
import torch
import numpy as np
import argparse as ap

from monai.data import *
from monai.metrics import *
from monai.transforms import Activations

from core.call_data import *
from core.call_loss import *
from core.call_model import *
from core.core import train
from config.call import call_config
from batchgenerators.utilities.file_and_folder_operations import *

def config_update(config):
    config["CONTRAST"] = [config["CONTRAST_L"],config["CONTRAST_U"]]
    if config["ISOTROPIC"]:
        config["INPUT_SHAPE"] = [config["INPUT_SHAPE_XY"],config["INPUT_SHAPE_XY"],config["INPUT_SHAPE_XY"]]
    else:
        config["INPUT_SHAPE"] = [config["INPUT_SHAPE_XY"],config["INPUT_SHAPE_XY"],config["INPUT_SHAPE_Z"]]
    
    if config["SPACING_XY"] is not None and config["SPACING_Z"] is not None:
        if config["ISOTROPIC"]:
            config["SPACING"] = [config["SPACING_XY"],config["SPACING_XY"],config["SPACING_XY"]]
        else:
            config["SPACING"] = [config["SPACING_XY"],config["SPACING_XY"],config["SPACING_Z"]]
    else:
        config["SPACING"] = None
    return config

def validation(input_shape, num_classes, val_loader, model, threshold=0.5):
    activation = Activations(sigmoid=True) # softmax : odd result! ToDO : check!
    dice_metric = DiceMetric(include_background=False, reduction='none')

    dice_class = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            step += 1
            val_inputs, val_labels = batch["image"].to("cuda", non_blocking=True), batch["label"].to("cuda", non_blocking=True).long()
            
            val_outputs = activation(sliding_window_inference(val_inputs, input_shape, 4, model))

            dice_class.append(dice_metric(val_outputs>=threshold, val_labels)[0])
    dice_val = 0.
    for class_ in num_classes:
        mean_ = 0
        for i in range(len(dice_class)):
            mean_ += dice_class[i][class_]
        dice_val += mean_ / len(dice_class)
    dice_val /= len(num_classes)

    torch.cuda.empty_cache()
    return dice_val

def data_load(data_dir, fold, num_folds, config):    
    # Dataset
    datasets = os.path.join(data_dir, 'dataset_monai.json')
    file_list = load_decathlon_datalist(datasets, True, 'training')
    train_list, valid_list = call_fold_dataset(file_list, target_fold=fold, total_folds=num_folds)    

    train_transforms, val_transforms = call_transforms(config)

    train_loader = call_dataloader(config, train_list, train_transforms, shuffle=True)
    valid_loader = call_dataloader(config, valid_list, val_transforms, shuffle=False)
    return train_loader, valid_loader

def run(config, info=None):
    torch.manual_seed(config["SEEDS"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["SEEDS"])

    config = config_update(config)

    train_loader, valid_loader = data_load(info["ROOT"], info["FOLD"], info["FOLDS"], config)

    model = call_model(config)
    if torch.cuda.device_count() > 1:
        print("Multi GPU activated!")
        model = nn.DataParallel(model)
    model.to("cuda")
    optimizer = call_optimizer(config, model)

    loss_function = call_loss(loss_mode = config["LOSS_NAME"], sigmoid=True)
    dice_loss = call_loss(loss_mode = 'dice', sigmoid=True)

    best_loss = 1.
    global_step = 0
    dice_val, dice_val_best = 0.0, 0.0
    while global_step < config["MAX_ITERATIONS"]:
        epoch_loss = 0
        epoch_dice = 0
        for step, batch in enumerate(train_loader):
            step += 1

            x, y = batch["image"].to("cuda", non_blocking=True), batch["label"].to("cuda", non_blocking=True).long()

            logit_map = model(x)

            loss = loss_function(logit_map, y)
            dice = 1 - dice_loss(logit_map, y)

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if (
                global_step % config["EVAL_NUM"] == 0 and global_step != 0
            ) or global_step == config["MAX_ITERATIONS"]:

                dice_val = validation(config["INPUT_SHAPE"], info["CHANNEL_OUT"], valid_loader, model, threshold=0.5)

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
        tune.report(
            loss=(epoch_loss / step), 
            dice=(epoch_dice / step), 
            dice_val=dice_val, 
            dice_val_best=dice_val_best, 
            global_step=global_step
        )
    return dice_val_best


def main_search(iterations, info, search):

    scheduler = ASHAScheduler(
        metric="dice_val_best",
        mode="max",
        max_t=iterations,
        grace_period=1,
        reduction_factor=2
    )
    algorithm = HyperOptSearch(
        metric="dice_val_best",
        mode="max"
    )
    reporter = CLIReporter(
        metric_columns=["loss", "dice", "dice_val", "global_step"]
    )

    result = tune.run(
        tune.with_parameters(run, info=info),
        config=search,
        search_alg=algorithm,
        resources_per_trial={"gpu":1, "cpu":8},
        num_samples=1000,
        scheduler=scheduler,  
        progress_reporter=reporter,
        local_dir='/nas3/jepark/ray_results'
    )
    best_trial = result.get_best_trial("dice_val_best", "max", "last")
    save_json(best_trial.config, join('/nas3/jepark/ray_results','{}_{}.json'.format(info["TARGET_NAME"],search["MODEL_NAME"])))