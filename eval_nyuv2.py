#!/usr/bin/env python3

import numpy as np
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import configargparse
from pdb import set_trace
from tqdm import tqdm
from pathlib import Path

from .metrics import get_depth_metrics
from .models.dorn import DORN
from .data.nyu_depth_v2.nyuv2_dataset import NYUDepthv2, NYUDepthv2Transient, NYUV2_CROP

from .experiment import ex

@ex.config('eval')
def cfg():
    parser = configargparse.ArgParser(default_config_files=[Path(__file__).parent/"eval.yml"],
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True)
    parser.add('--mde', required=True, help="MDE to evaluate.")
    parser.add('--augment', choices=['median', 'gt_hist', 'transient'], help='Optional augmenting')
    parser.add('--sbr', type=float, help='sbr for transient')
    parser.add('--split', choices=['train', 'test'], required=True)
    parser.add('--transform', action='append')
    config, _ = parser.parse_known_args()
    set_trace()
    return vars(config)


@ex.entity('NYUv2Evaluation')
class NYUv2Evaluation:
    def __init__(self, model, dataset, crop=NYUV2_CROP):
        self.model = model
        self.dataset = dataset
        self.crop = crop

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=1)
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


def get_mde(config):
    if config['mde'] == 'DORN':
        from .models.dorn import DORN
        dorn_config = ex.config['DORN']
        mde = DORN(**dorn_config)
        model = lambda data: mde(data['dorn_image'], resize=(640, 480))
        return model
    elif config['mde'] == 'DenseDepth':
        raise NotImplementedError
    elif args.model == 'MiDaS':
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown model: {args.model}")


def get_dataset():
    if config['augment'] in [None, 'median', 'gt_hist']:
        transforms.append(ex.transforms['crop_image_and_depth'])
        dataset = NYUDepthv2(split=config['split'])
    elif config['augment'] == 'transient':
        transient_config = ex.configs['transient']
        transforms.append(ex.transforms['crop_image_and_depth'])
        dataset = NYUDepthv2Transient(split=config['split'], sbr=config['sbr'])
        preproc = TransientPreprocessor(n_sid_bins=transient_config['n_sid_bins'],
                                        n_ambient_bins=transient_config['ambient_bins'],
                                        beta=transient_config['beta'],
                                        n_std=transient_config['n_std'])
    else:
        raise ValueError(f"Unknown method: {config['augment']}")


if __name__ == '__main__':
    set_trace()
    mde = get_mde()
    dataset = get_dataset()
    transform = Compose([ex.transforms[t] for t in ex.configs['eval']['transform']])
    dataset.transform = transform
    evaluator = NYUv2Evaluation(model, dataset)
    set_trace()
    metrics = evaluator.evaluate()
