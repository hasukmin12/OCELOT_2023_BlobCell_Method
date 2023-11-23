import torch
import wandb

def temp_image(info, frame, image, label, pred):
    # if info["VISUAL_AXIS"] == 1:
    #     return image[frame].numpy(), label[frame].numpy(), pred[frame].numpy()
    # if info["VISUAL_AXIS"] == 2:
    #     return image[:,frame].numpy(), label[:,frame].numpy(), pred[:,frame].numpy()
    # if info["VISUAL_AXIS"] == 3:
    #     return image[...,frame].numpy(), label[...,frame].numpy(), pred[...,frame].numpy()
    return image.numpy(), label.numpy(), pred.numpy()

def log_image_table(info, image, label, predict):
    mask_images = []
    image = image[0]
    # predict = predict[0] # 추가함
    label = torch.argmax(label, dim=0) if not label.size()[0] == 1 else label[0]
    predict = torch.argmax(predict, dim=0) if not predict.size()[0] == 1 else predict[0]

    frames = int(round(image.shape[info["VISUAL_AXIS"]-1]/3))
    for frame in range(frames,frames*2,2):
        t_image, t_label, t_pred = temp_image(info, frame, image, label, predict)
        mask_images.append(wandb.Image(t_image, masks={
            "ground_truth":{"mask_data":t_label,"class_labels":info["CLASS_NAMES"]},
            "predictions":{"mask_data":t_pred,"class_labels":info["CLASS_NAMES"]},
        }))
    return mask_images

def calc_mean_class(info, list_, metric_class='valid_dice'):
    all_mean = {}
    all_value = 0
    except_0 = 0
    for class_ in range(info["CHANNEL_OUT"]-1):
        mean_ = 0
        cnt = 0
        for i in range(len(list_)):
            # x = list_[i][class_]
            # y = torch.isfinite(x)
            # z = torch.isnan(x)
            if torch.isnan(list_[i][class_]) == False:
                mean_ += list_[i][class_]
                cnt += 1
        if cnt != 0:
            all_mean.update({f'{metric_class}/{info["CLASS_NAMES"][class_+1]}':(mean_/len(list_)).item()})
            all_value += mean_/cnt
        else:
            all_mean.update({f'{metric_class}/{info["CLASS_NAMES"][class_+1]}':(mean_/len(list_))})
            all_value += mean_
            except_0 += 1

    all_value /= ((info["CHANNEL_OUT"]-1) - except_0)
    # # added
    # all_value = all_value.item()
    return all_mean, all_value



# def calc_mean_class(info, list_, metric_class='valid_dice'):
#     all_mean = {}
#     all_value = 0
#     for class_ in range(info["CHANNEL_OUT"]-1):
#         mean_ = 0
#         for i in range(len(list_)):
#             mean_ += list_[i][class_]
#         all_mean.update({f'{metric_class}/{info["CLASS_NAMES"][class_+1]}':(mean_/len(list_)).item()})
#         all_value += mean_/len(list_)
#     all_value /= (info["CHANNEL_OUT"]-1)
#     # # added
#     # all_value = all_value.item()
#     return all_mean, all_value





def calc_confusion_metric(metric_name, confusion_matrix):
    tp, fp, tn, fn = confusion_matrix[0],confusion_matrix[1],confusion_matrix[2],confusion_matrix[3]
    p = tp + fn
    n = fp + tn
    # calculate metric
    metric = check_confusion_matrix_metric_name(metric_name)
    numerator: torch.Tensor
    denominator: Union[torch.Tensor, float]
    nan_tensor = torch.tensor(float("nan"), device=confusion_matrix.device)
    if metric == "tpr":
        numerator, denominator = tp, p
    elif metric == "tnr":
        numerator, denominator = tn, n
    elif metric == "ppv":
        numerator, denominator = tp, (tp + fp)
    elif metric == "npv":
        numerator, denominator = tn, (tn + fn)
    elif metric == "fnr":
        numerator, denominator = fn, p
    elif metric == "fpr":
        numerator, denominator = fp, n
    elif metric == "fdr":
        numerator, denominator = fp, (fp + tp)
    elif metric == "for":
        numerator, denominator = fn, (fn + tn)
    elif metric == "pt":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
        denominator = tpr + tnr - 1.0
    elif metric == "ts":
        numerator, denominator = tp, (tp + fn + fp)
    elif metric == "acc":
        numerator, denominator = (tp + tn), (p + n)
    elif metric == "ba":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator, denominator = (tpr + tnr), 2.0
    elif metric == "f1":
        numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
    elif metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elif metric == "fm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        numerator = torch.sqrt(ppv * tpr)
        denominator = 1.0
    elif metric == "bm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = tpr + tnr - 1.0
        denominator = 1.0
    elif metric == "mk":
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        npv = torch.where((tn + fn) > 0, tn / (tn + fn), nan_tensor)
        numerator = ppv + npv - 1.0
        denominator = 1.0
    else:
        raise NotImplementedError("the metric is not implemented.")

    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator != 0, numerator / denominator, nan_tensor)
    return numerator / denominator

def check_confusion_matrix_metric_name(metric_name: str):
    """
    There are many metrics related to confusion matrix, and some of the metrics have
    more than one names. In addition, some of the names are very long.
    Therefore, this function is used to check and simplify the name.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
        return "tpr"
    if metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
        return "tnr"
    if metric_name in ["precision", "positive_predictive_value", "ppv"]:
        return "ppv"
    if metric_name in ["negative_predictive_value", "npv"]:
        return "npv"
    if metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
        return "fnr"
    if metric_name in ["fall_out", "false_positive_rate", "fpr"]:
        return "fpr"
    if metric_name in ["false_discovery_rate", "fdr"]:
        return "fdr"
    if metric_name in ["false_omission_rate", "for"]:
        return "for"
    if metric_name in ["prevalence_threshold", "pt"]:
        return "pt"
    if metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
        return "ts"
    if metric_name in ["accuracy", "acc"]:
        return "acc"
    if metric_name in ["balanced_accuracy", "ba"]:
        return "ba"
    if metric_name in ["f1_score", "f1"]:
        return "f1"
    if metric_name in ["matthews_correlation_coefficient", "mcc"]:
        return "mcc"
    if metric_name in ["fowlkes_mallows_index", "fm"]:
        return "fm"
    if metric_name in ["informedness", "bookmaker_informedness", "bm"]:
        return "bm"
    if metric_name in ["markedness", "deltap", "mk"]:
        return "mk"
    raise NotImplementedError("the metric is not implemented.")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
