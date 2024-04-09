"""Metric 함수 정의
"""

import torch
import numpy as np
from torchmetrics import F1Score, Accuracy

SMOOTH = 1e-6

def get_metric_function(metric_function_str,device):
    """
    Add metrics, weights for weighted score
    """

    if metric_function_str == 'f1-score':
        return F1Score(num_classes=4, average='weighted',task='multiclass').to(device)
    elif metric_function_str == 'acc':
        return Accuracy(task="multiclass", num_classes=4).to(device)
