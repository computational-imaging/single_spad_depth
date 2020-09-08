#!/usr/bin/env python3

import configargparse
configargparse.get_argument_parser(
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    add_help=True
)
import torch
import numpy as np
from pdb import set_trace
from pathlib import Path
from torchvision.transforms import Compose

from core.metrics import get_depth_metrics
from core.experiment import ex
from models.mde_transient import MDETransient
from data.nyu_depth_v2.nyuv2_dataset import to_tensor
from torch.utils.data._utils.collate import default_collate


IN_DATA_DIR = Path(__file__).parent/'data'/'captured'/'processed'

@ex.add_arguments('CapturedEvaluation')
def cfg():
    parser = configargparse.get_argument_parser()
    group = parser.add_argument_group('eval_captured', 'evaluation params.')
    group.add('--scene-config', is_config_file=True)
    group.add('--mde-config', is_config_file=True)
    group.add('--method', choices=['mde', 'transient'], required=True)
    group.add('--scene', choices=['kitchen', 'classroom', 'desk', 'hallway',
                                  'poster', 'lab', 'outdoor'],
              default='lab')
    group.add('--transform', action='append')
    group.add('--output-dir', default=str(Path(__file__).parent/'results_captured'))
    group.add('--gpu', type=str)

@ex.setup('CapturedEvaluation')
def setup(config):
    scene_data = np.load(IN_DATA_DIR/f'{config["scene"]}.npy', allow_pickle=True)[()]
    model = ex.get_and_configure(config['method'])
    transform = Compose([ex.transforms[t] for t in config['transform']])
    if config['gpu'] is not None and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        device = torch.device('cuda')
        print(f"Using gpu {config['gpu']} (CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}).")
    else:
        device = torch.device('cpu')
        print("Using cpu.")
    return CapturedEvaluation(model=model, scene_data=scene_data, transform=transform, device=device)


@ex.entity
class CapturedEvaluation:
    def __init__(self, model, scene_data, transform, device):
        self.model = model
        self.scene_data = scene_data
        self.transform = transform
        self.device = device

    def evaluate(self):
        # Load preprocessed data:
        data = self.transform(self.scene_data)
        data = default_collate([data])
        for k, v in data.items():
            data[k] = v.to(self.device)
        depth = self.model(data)
        pred = {'depth': depth}
        pred['metrics'] = self.compute_metrics(data, pred)
        return pred

    def compute_metrics(self, data, pred):
        p = pred['depth'].cpu().squeeze()
        d = data['depth'].cpu().squeeze()
        m = data['mask'].cpu().squeeze()
        metrics = get_depth_metrics(p, d, m)
        return metrics


if __name__ == '__main__':
    evaluator = ex.get_and_configure('CapturedEvaluation')
    pred = evaluator.evaluate()
    pred['depth'] = pred['depth'].numpy()
    scene = ex.config['scene']
    mde_name = ex.config['mde']
    output_dir = Path(ex.config['output_dir'])/f'{scene}'/f'{mde_name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir/f'{ex.config["method"]}.npy', pred)
