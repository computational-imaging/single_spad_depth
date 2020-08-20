#!/usr/bin/env python3

import numpy as np
import torch
import configargparse
from .data.nyu_depth_v2.dataloader import get_dataloader, NYUV2_CROP
from .metrics import get_depth_metrics
from pdb import set_trace
from tqdm import tqdm


from .registry import REGISTRY

parser = configargparse.ArgParser(default_config_files=['eval.cfg'])
parser.add('-c', is_config_file=True)
parser.add('--model', required=True, help="Model to evaluate.")
parser.add('--split', required=True)
parser.add('--spad-file')


class NYUv2Evaluation:
    def __init__(self, model, split, transform, crop=NYUV2_CROP):
        self.model = model
        self.split = split
        self.transform = transform
        self.crop = crop

    def evaluate(self):
        dataloader = get_dataloader(self.split, self.transform)
        preds = []
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            depth = self.model(data)
            pred = {'depth': depth}
            if crop is not None:
                depth_cropped = pred[..., crop[0]:crop[1], crop[2]:crop[3]]
                pred['depth_cropped'] = depth_cropped
            pred['metrics'] = self.compute_metrics(data, pred)
            preds.append(pred)
        return metrics

    def compute_metrics(self, data, pred):
        metrics = get_depth_metrics(pred['depth_cropped'], data['depth_cropped'],
                                    torch.ones_like(pred['depth_cropped']))
        return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    if args.model == "DORN":
        from .mde.dorn import DORN, dorn_transform
        mde = DORN()
        transform = dorn_transform
    elif args.model == "DenseDepth":
        raise NotImplementedError
    elif args.model == "MiDaS":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown model: {args.model}")

    evaluator = NYUv2Evaluation(model, args.split, transform)
    set_trace()
    metrics = evaluator.evaluate()
