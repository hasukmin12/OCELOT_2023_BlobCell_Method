import os, sys, glob
sys.path.append('../')
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm

from monai.data import *
from monai.metrics import *
from monai.transforms import Activations
from monai.inferers import sliding_window_inference

from core.utils import *   
from core.call_loss import *
from core.call_model import *
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
import monai
from monai.networks import one_hot

from core.metric.metrics import *




def validation(info, config, valid_loader, model, logging=False, threshold=0.5):  
    activation = Activations(sigmoid=True) # softmax : odd result! ToDO : check!  
    dice_metric = DiceMetric(include_background=False, reduction='none')
    confusion_matrix = ConfusionMatrixMetric(include_background=False, reduction='none')
    # dice_metric = dice()
    # prec = precision()
    # rec = recall()
    # f1s = fscore()
    # h_d = hausdorff_distance()
    # confusion_matrix = ConfusionMatrix()

    epoch_iterator_val = tqdm(
        valid_loader, desc="Validate (X / X Steps)", dynamic_ncols=True
    )
    dice_class, mr_class, fo_class = [], [], []
    real_dice, prec_class, rec_class, f1_class, h_d_class = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            step += 1
            
            val_inputs, val_labels = batch["image"].to("cuda"), batch["label"].to("cuda") # , non_blocking=True).long()

            val_labels = one_hot(val_labels, config["CHANNEL_OUT"])
            val_outputs = sliding_window_inference(val_inputs, config["INPUT_SHAPE"], 4 , model) # , device='cuda', sw_device='cuda')
            # val_outputs = val_outputs[0]
            
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (step, len(epoch_iterator_val))
            )

            if info["Deep_Supervision"] == True: 
                dice_class.append(dice(val_outputs[0]>=threshold, val_labels)[0])

                confusion = confusion_matrix(val_outputs[0]>=threshold, val_labels)[0]
                mr_class.append([
                    calc_confusion_metric('fnr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])
                fo_class.append([
                    calc_confusion_metric('fpr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])

                # prec_class.append(precision(val_outputs[0]>=threshold, val_labels)[0])
                # rec_class.append(rec(val_outputs[0]>=threshold, val_labels)[0])
                # f1_class.append(f1s(val_outputs[0]>=threshold, val_labels)[0])
                # h_d_class.append(h_d(val_outputs[0]>=threshold, val_labels)[0])

                test_pred_out = torch.nn.functional.softmax(val_outputs[0], dim=1) # (B, C, H, W)
                test_pred_out = torch.argmax(test_pred_out[0], axis=0)
                label_out = torch.argmax(val_labels[0], axis=0)

                real_dice.append(dice(test_pred_out, label_out))
                prec_class.append(precision(test_pred_out, label_out))
                rec_class.append(recall(test_pred_out, label_out))
                f1_class.append(fscore(test_pred_out, label_out))




            else:
                # conf_metrics = ConfusionMatrix(val_outputs>=threshold, val_labels)
                # rst = dice(val_outputs>=threshold, val_labels)
                # rst2 = DiceMetric(val_outputs>=threshold, val_labels)
                dice_class.append(dice_metric(val_outputs>=threshold, val_labels)[0])
                confusion = confusion_matrix(val_outputs>=threshold, val_labels)[0]
                mr_class.append([
                    calc_confusion_metric('fnr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])
                fo_class.append([
                    calc_confusion_metric('fpr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])

                test_pred_out = torch.nn.functional.softmax(val_outputs, dim=1) # (B, C, H, W)
                test_pred_out = torch.argmax(test_pred_out[0], axis=0)
                label_out = torch.argmax(val_labels[0], axis=0)

                real_dice.append(dice(test_pred_out, label_out))
                prec_class.append(precision(test_pred_out, label_out))
                rec_class.append(recall(test_pred_out, label_out))
                f1_class.append(fscore(test_pred_out, label_out))
                # h_d_class.append(hausdorff_distance(test_pred_out, val_labels))

            # torch.cuda.empty_cache()
        dice_dict, dice_val_class = calc_mean_class(info, dice_class, 'valid_dice')
        miss_dict, miss_val = calc_mean_class(info, mr_class, 'valid_miss rate')
        Val_dice = np.mean(real_dice)
        Prec = np.mean(prec_class)
        Recall = np.mean(rec_class)
        F1_c = np.mean(f1_class)

        # prec_dict, prec_val = calc_mean_class(info, prec_class, 'valid_precision')
        # rec_dict, rec_val = calc_mean_class(info, rec_class, 'valid_recall')
        # f1_dict, f1_val = calc_mean_class(info, f1_class, 'valid_F1_score')
        # h_d_dict, h_d_val = calc_mean_class(info, h_d_class, 'valid_hausdorff_distance')
        # false_dict, false_val = calc_mean_class(info, fo_class, 'valid_false alarm')
        if info["Deep_Supervision"]==True:
            # print(dice_val)
            # print(dice_val.item())
            wandb.log({'valid_dice': dice_val_class.item(), 'valid_miss rate': miss_val.item(),
                       'Real_dice' : Val_dice, 'f1_score' : F1_c,
                       'precision' : Prec, 'recall': Recall,
            # 'valid_image': log_image_table(info, val_inputs[0].cpu(),
                                            # val_labels[0].cpu(), val_outputs[0][0].cpu())
                                            })
            wandb.log(dice_dict)
            
        else:
            # print(dice_val)
            # print(dice_val.item())
            wandb.log({'valid_dice': dice_val_class.item(), 'valid_miss rate': miss_val.item(),
                       'Real_dice' : Val_dice, 'f1_score' : F1_c,
                       'precision' : Prec, 'recall': Recall,
            # 'valid_image': log_image_table(info, val_inputs[0].cpu(),
            #                                 val_labels[0].cpu(), val_outputs[0].cpu())
            })                            
        # })

            wandb.log(dice_dict)
            # wandb.log(miss_dict)
            # wandb.log(false_dict)        
    return Val_dice, dice_val_class



def train(info, config, global_step, dice_val_best, steps_val_best, dice_val_best_class, steps_val_best_class, model, optimizer, train_loader, valid_loader, logging=False, deep_supervision=True): 
    # print(model)
    loss_function = call_loss(loss_mode=config["LOSS_NAME"], sigmoid=True, config=config)
    dice_loss_f = call_loss(loss_mode='dice', sigmoid=True)
    # dicece_loss = call_loss(loss_mode='dicece', sigmoid=True)
    ce_loss_f = call_loss(loss_mode='ce', sigmoid=True)
 
    model.train()

    step = 0
    epoch_loss, epoch_dice = 0., 0.
    print(train_loader)
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    print(epoch_iterator)
    for step, batch in enumerate(epoch_iterator):
        step += 1

        x, y1 = batch["image"].to("cuda"), batch["label"].to("cuda") # .long()
        # x, y = batch["image"].to("cuda", non_blocking=True), batch["label"].to("cuda", non_blocking=True).long()
        # x = x.squeeze(dim=4)
        # x = x.view([-1, 1, 196, 196, 1])
        # print(model)
        logit_map = model(x)

        y = one_hot(
                y1, config["CHANNEL_OUT"]
            )  # (b,cls,256,256)

        loss = 0
        if deep_supervision == True: # deep supervision
            for ds in logit_map:
                loss += loss_function(ds, y)
                # ce_loss = ce_loss_f(ds, y)
                # dice_loss = dice_loss_f(ds, y)
                # dice += 1 - dice_loss
                # loss += ce_loss + dice_loss

            loss /= len(logit_map)
            # dice /= len(logit_map)
        else:
            loss = loss_function(logit_map, y)
            # ce_loss = ce_loss_f(logit_map, y)
            # dice_loss = dice_loss_f(logit_map, y)
            # dice = 1 - dice_loss
            # loss = ce_loss + dice_loss

        # print(loss)
        epoch_loss += loss.item()
        # epoch_dice += dice.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step+1, config["MAX_ITERATIONS"], loss)
        )

        # 0.5k마다 model save
        if global_step % 500 == 0 and global_step > 500 :
            torch.save({
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(info["LOGDIR"], "model_e{0:05d}.pth".format(global_step)))
            print(
                f"Model Was Saved ! steps: {global_step}"
            )


        if (
            global_step % config["EVAL_NUM"] == 0 and global_step != 0
        ) or global_step == config["MAX_ITERATIONS"]:
            
            dice_val, dice_val_class = validation(info, config, valid_loader, model, logging)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                steps_val_best = global_step
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(info["LOGDIR"], "model_best_e{0:05d}.pth".format(global_step)))
                print(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )
            else:
                print(
                    f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val} Best model step is {steps_val_best}" 
                )



            # for dice_val_class
            if dice_val_class > dice_val_best_class:
                dice_val_best_class = dice_val_class
                steps_val_best_class = global_step
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(info["LOGDIR"], "class_model_best_e{0:05d}.pth".format(global_step)))
                print(
                    f"(Class Dice) Model Was Saved ! Current Best Avg. Dice: {dice_val_best_class} Current Avg. Dice: {dice_val_class}"
                )
            else:
                print(
                    f"(Class Dice) Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best_class} Current Avg. Dice: {dice_val_class} Best model step is {steps_val_best_class}" 
                )


            # # 1k마다 model save
            # if global_step % 525 == 0 and global_step != 0 :
            #     torch.save({
            #         "global_step": global_step,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #     }, os.path.join(info["LOGDIR"], "model_e{0:05d}.pth".format(global_step)))
            #     print(
            #         f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
            #     )

        global_step += 1
    return global_step, dice_val_best, steps_val_best, dice_val_best_class, steps_val_best_class, epoch_loss / step # , epoch_dice / step
