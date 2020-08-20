#!/usr/bin/env python3

import torch

def get_depth_metrics(depth_pred, depth_truth, mask):
    """
    Takes torch tensors.
    :param depth_pred: Depth prediction
    :param depth_truth: Ground truth
    :param mask: Masks off invalid pixels
    :return: Dictionary of metrics
    """
    metrics = dict()
    # deltas
    metrics["delta1"] = delta(depth_pred, depth_truth, mask, 1.25).item()
    metrics["delta2"] = delta(depth_pred, depth_truth, mask, 1.25 ** 2).item()
    metrics["delta3"] = delta(depth_pred, depth_truth, mask, 1.25 ** 3).item()
    # rel_abs_diff
    metrics["rel_abs_diff"] = rel_abs_diff(depth_pred, depth_truth, mask).item()
    # rel_sqr_diff
    metrics["rel_sqr_diff"] = rel_sqr_diff(depth_pred, depth_truth, mask).item()
    # log10
    metrics["log10"] = log10(depth_pred, depth_truth, mask).item()
    # mse
    metrics["mse"] = mse(depth_pred, depth_truth, mask).item()
    # rmse
    metrics["rmse"] = rmse(depth_pred, depth_truth, mask).item()
    # rmse(log)
    metrics["log_rmse"] = rmse(torch.log(depth_pred),
                               torch.log(depth_truth),
                               mask).item()
    # print(metrics)
    return metrics

def delta(prediction, target, mask, threshold):
    """
    Given prediction and target, compute the fraction of indices i
    such that
    max(prediction[i]/target[i], target[i]/prediction[i]) < threshold
    """
    # print(prediction.dtype)
    # print(mask.dtype)
    # print(target.dtype)
    c = torch.max(prediction[mask > 0]/target[mask > 0], target[mask > 0]/prediction[mask > 0])
    return torch.sum((c < threshold).float())/(torch.sum(mask))

def mse(prediction, target, mask):
    """
    Return the MSE of prediction and target
    """
    diff = prediction - target
    squares = (diff[mask > 0]).pow(2)
    out = torch.sum(squares)
    total = torch.sum(mask).item()
    if total > 0:
        return (1. / torch.sum(mask)) * out
    else:
        return torch.zeros(1)

def rmse(prediction, target, mask):
    """
    Return the RMSE of prediction and target
    """
    return torch.sqrt(mse(prediction, target, mask))


def log10(prediction, target, mask, size_average=True):
    """
    Return the log10 loss metric:
    1/N * sum(| log_10(prediction) - log_10(target) |)
    :param prediction:
    :param target:
    :param mask:
    :return:
    """
    out = torch.sum(torch.abs(torch.log10(prediction[mask > 0]) - torch.log10(target[mask > 0])))
    if size_average:
        total = torch.sum(mask).item()
        if total > 0:
            return (1./torch.sum(mask))*out
        else:
            return torch.zeros(1)
    return out

def rel_abs_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative absolute difference:

    1/N*sum(|prediction - target|/target)
    """
    diff = prediction - target
    out = torch.sum(torch.abs(diff[mask > 0])/(target[mask > 0] + eps))
    total = torch.sum(mask).item()
    if total > 0:
        return (1. / torch.sum(mask)) * out
    else:
        return torch.zeros(1)


def rel_sqr_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative squared difference:

    1/N*sum(||prediction - target||**2/target)
    """
    diff = prediction - target
    out = torch.sum((diff[mask > 0]).pow(2)/(target[mask > 0] + eps))
    total = torch.sum(mask).item()
    if total > 0:
        return (1. / torch.sum(mask)) * out
    else:
        return torch.zeros(1)
