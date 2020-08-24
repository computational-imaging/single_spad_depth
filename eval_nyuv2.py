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
# Models
from .models.mde import MDE
from .models.mde_median import MDEMedian
from .models.mde_transient import MDETransient

# Datasets
from .data.nyu_depth_v2.nyuv2_dataset import NYUDepthv2, NYUDepthv2Transient, NYUV2_CROP

from .experiment import ex

@ex.config('NYUv2Evaluation')
def cfg():
    parser = configargparse.ArgParser(default_config_files=[str(Path(__file__).parent/'eval.yml')],
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True)
    parser.add('--model', required=True, help="Model to evaluate.")
    parser.add('--dataset', choices=['normal', 'transient'], help='Which dataset to use.')
    parser.add('--sbr', type=float, help='sbr for transient')
    parser.add('--split', choices=['train', 'test'], required=True)
    parser.add('--transform', action='append')
    parser.add('--output-dir', default=str(Path(__file__).parent/'results'))
    config, _ = parser.parse_known_args()
    # set_trace()
    return vars(config)

@ex.setup('NYUv2Evaluation')
def setup(config):
    model = ex.get_and_configure(config['model'])
    if config['dataset'] == 'normal':
        dataset = NYUDepthv2(split=config['split'])
    else:
        dataset = NYUDepthv2Transient(split=config['split'], sbr=config['sbr'])
    transform = Compose([ex.transforms[t] for t in config['transform']])
    dataset.transform = transform
    return NYUv2Evaluation(model, dataset)

@ex.entity
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
            if self.crop is not None:
                pred['depth_cropped'] = depth[...,
                                              self.crop[0]:self.crop[1],
                                              self.crop[2]:self.crop[3]]
            pred['metrics'] = self.compute_metrics(data, pred)
            preds.append(pred)
            # set_trace()
            # print(pred['metrics'])
            # DEBUG
            # if i == 1:
                # break
        return preds

    def compute_metrics(self, data, pred):
        p = pred['depth_cropped'].squeeze()
        d = data['depth_cropped'].squeeze()
        m = torch.ones_like(d)
        metrics = get_depth_metrics(p, d, m)
        return metrics

def summarize(all_metric_dicts):
    """Metrics is a list of dictionaries, each with <metric>: value entries"""
    if len(all_metric_dicts) == 0:
        return {}
    summary = {}
    metric_names = all_metric_dicts[0].keys()
    for k in metric_names:
        summary[k] = np.mean([metric_dict[k] for metric_dict in all_metric_dicts])
    return summary

if __name__ == '__main__':
    evaluator = ex.get_and_configure('NYUv2Evaluation')
    preds = evaluator.evaluate()
    summary = summarize([p['metrics'] for p in preds])
    config = cfg()
    model_name = config['model']
    mde_name = evaluator.model.mde.__class__.__name__
    output_dir = Path(config['output_dir'])/f'{model_name}'/f'{mde_name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir/'summary', summary)
