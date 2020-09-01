#!/usr/bin/env python3

import configargparse

from core.experiment import ex

@ex.entity
class CapturedEvaluation:
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=1)
        preds = []
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            for k, v in data.items():
                data[k] = v.to(self.device)
            depth = self.model(data)
            pred = {'z_pred': depth}
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
        p = pred['z_pred'].cpu().squeeze()
        d = data['z_gt'].cpu().squeeze()
        m = data['mask']
        metrics = get_depth_metrics(p, d, m)
        return metrics




if __name__ == '__main__':
    parser = configargparse.get_arg_parser()
    ex.config = parser.parse_args()
    evaluator = ex.get_and_configure('CapturedEvaluation')
    evaluator.evaluate()
