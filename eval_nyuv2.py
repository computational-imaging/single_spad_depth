#!/usr/bin/env python3

import numpy as np
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
# Initialize singleton parser
import configargparse
configargparse.get_argument_parser(
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    add_help=True)
from pdb import set_trace
from tqdm import tqdm
from pathlib import Path
import os

from core.metrics import get_depth_metrics
# Models
from models.mde import MDE
from models.mde_median import MDEMedian
from models.mde_gt_hist import MDEGTHist
from models.mde_transient import MDETransient

# Datasets
from data.nyu_depth_v2.nyuv2_dataset import NYUDepthv2, NYUDepthv2Transient, NYUV2_CROP

from core.experiment import ex

@ex.add_arguments('NYUv2Evaluation')
def cfg():
    parser = configargparse.get_argument_parser()
    group = parser.add_argument_group('eval_nyuv2', 'evaluation params.')
    group.add('-c', is_config_file=True)
    group.add('--method', choices=['mde', 'median', 'gt_hist', 'transient'],
               default='mde', help="Method to evaluate.")
    group.add('--sbr', type=float, help='sbr for transient method')
    group.add('--split', choices=['train', 'test'], default='test')
    group.add('--transform', action='append')
    group.add('--pre-cropped', action='store_true',
               help="True if the model being evaluated already outputs cropped depth images.")
    group.add('--output-dir', default=str(Path(__file__).parent/'results'))
    group.add('--gpu', type=str)
    # config, _ = parser.parse_known_args()
    # return vars(config)

@ex.setup('NYUv2Evaluation')
def setup(config):
    model = ex.get_and_configure(config['method'])
    if config['method'] in ['mde', 'median', 'gt_hist']:
        dataset = NYUDepthv2(split=config['split'])
    else:
        dataset = NYUDepthv2Transient(split=config['split'], sbr=config['sbr'])
    transform = Compose([ex.transforms[t] for t in config['transform']])
    dataset.transform = transform
    if config['gpu'] is not None and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        device = torch.device('cuda')
        print(f"Using gpu {config['gpu']} (CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}).")
    else:
        device = torch.device('cpu')
        print("Using cpu.")
    return NYUv2Evaluation(model=model,
                           dataset=dataset,
                           device=device,
                           pre_cropped=config['pre_cropped'])

@ex.entity
class NYUv2Evaluation:
    def __init__(self, model, dataset, device, crop=NYUV2_CROP, pre_cropped=False):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.crop = crop
        self.pre_cropped = pre_cropped

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=1)
        preds = []
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            for k, v in data.items():
                data[k] = v.to(self.device)
            depth = self.model(data)
            pred = {'depth': depth}
            if self.pre_cropped:
                pred['depth_cropped'] = depth
            else:
                pred['depth_cropped'] = depth[...,
                                              self.crop[0]:self.crop[1],
                                              self.crop[2]:self.crop[3]]
            pred['metrics'] = self.compute_metrics(data, pred)
            preds.append(pred)
            # set_trace()
            # print(pred['metrics'])
            # DEBUG
            # if i == 0:
                # print(pred['metrics'])
                # break
        return preds

    def compute_metrics(self, data, pred):
        p = pred['depth_cropped'].cpu().squeeze()
        d = data['depth_cropped'].cpu().squeeze()
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
    # parser = configargparse.get_arg_parser()
    # ex.config = vars(parser.parse_args())
    # set_trace()
    evaluator = ex.get_and_configure('NYUv2Evaluation')
    # set_trace()
    print(f"Evaluating {ex.config['mde']} in " + \
          f"{ex.config['method']} mode.")
    preds = evaluator.evaluate()
    summary = summarize([p['metrics'] for p in preds])
    depth_preds_cropped = np.stack([p['depth_cropped'].cpu().squeeze() for p in preds], axis=0)
    config = cfg()
    method_name = ex.config['method']
    mde_name = ex.config['mde']
    output_dir = Path(ex.config['output_dir'])/f'{method_name}'/f'{mde_name}'
    if ex.config['method'] == 'transient':
        output_dir = output_dir/f'sbr_{ex.config["sbr"]}'
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir/'summary', summary)
    np.save(output_dir/'preds_cropped', depth_preds_cropped)
