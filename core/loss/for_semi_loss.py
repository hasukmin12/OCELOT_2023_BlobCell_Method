import numpy
import torch
import torch.nn.functional as F



# softmax mse loss
def semi_mse_loss(pred, y_logits):
    y_prob = torch.softmax(y_logits, dim=1)
    pred_prob = torch.softmax(pred, dim=1)

    return F.mse_loss(y_prob, pred_prob, reduction='none')


def semi_softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice




def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss




class ConsistencyWeight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, len_loader):
        self.final_w = 1.0
        self.iter_per_epoch = len_loader
        self.start_iter = 0 * len_loader
        self.rampup_length = 40 * len_loader
        self.rampup_func = getattr(self, "sigmoid")
        self.current_rampup = 0

    def __call__(self, current_idx):
        if current_idx <= self.start_iter:
            return .0

        self.current_rampup = self.rampup_func(current_idx-self.start_iter,
                                               self.rampup_length)

        return self.final_w * self.current_rampup

    @staticmethod
    def gaussian(start, current, rampup_length):
        assert rampup_length >= 0
        if current == 0:
            return .0
        if current < start:
            return .0
        if current >= rampup_length:
            return 1.0
        return numpy.exp(-5 * (1 - current / rampup_length) ** 2)

    @staticmethod
    def sigmoid(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = numpy.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
    
        return float(numpy.exp(-5.0 * phase * phase))

    @staticmethod
    def linear(current, rampup_length):
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        return current / rampup_length



# peer-based binary cross-entropy
def semi_cbc_loss(inputs, targets,
                  threshold=0.6,
                  neg_threshold=0.3,
                  conf_mask=True):
    if not conf_mask:
        raise NotImplementedError
    targets_prob = F.softmax(targets, dim=1)
    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = (pos_weight >= threshold)
    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_mask = (neg_weight < neg_threshold)
    y_tilde = torch.argmax(targets, dim=1)
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(torch.log_softmax(inputs, dim=1),
                                       y_tilde, reduction="none")
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]
    
    if not torch.any(neg_mask):
        negative_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        inverse_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-6, max=1.0)
        negative_loss_mat = F.nll_loss(inverse_prob.log(), (1-y_tilde), reduction="none")
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]
    
    return positive_loss_mat.mean() + negative_loss_mat.mean() # , None









def semi_softmax_diceFocal_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice



def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss