import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from core.activation import NormReLU
from monai.transforms import Activations
"""
https://github.com/SeungyounShin/Adaptive-Wing-Loss-for-Robust-Face-Alignment-via-Heatmap-Regression/blob/master/losses/loss.py
"""
class AWing(nn.Module):

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5, activation='NormRelu'):
        super().__init__()
        self.alpha   = float(alpha)
        self.omega   = float(omega)
        self.epsilon = float(epsilon)
        self.theta   = float(theta)
        if activation in ['NormRelu', 'normrelu']:
            self.activation = NormReLU()
        elif activation in ['Sigmoid', 'sigmoid']:
            self.activation = Activations(sigmoid=True)
        elif activation in ['Softmax', 'softmax']:
            self.activation = Activations(softmax=True)


    def forward(self, y_pred , y):
        y_pred = self.activation(y_pred)

        # lossMat = torch.zeros_like(y_pred)
        # A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y)))*(self.alpha-y)*((self.theta/self.epsilon)**(self.alpha-y-1))/self.epsilon
        # C = self.theta*A - self.omega*torch.log(1+(self.theta/self.epsilon)**(self.alpha-y))
        # case1_ind = torch.abs(y-y_pred) < self.theta
        # case2_ind = torch.abs(y-y_pred) >= self.theta
        # lossMat[case1_ind] = self.omega*torch.log(1+torch.abs((y[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y[case1_ind]))
        # lossMat[case2_ind] = A[case2_ind]*torch.abs(y[case2_ind]-y_pred[case2_ind]) - C[case2_ind]
        # if self.do_mean:
        #     return torch.mean(lossMat)
        #     # return (lossMat[case1_ind].sum()+lossMat[case2_ind].sum())/(len(case1_ind)+len(case2_ind))
        # else:
        #     return lossMat

        delta_y = (y - y_pred).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))