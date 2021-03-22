import argparse
import sys
sys.path.append('/home/billyhe/SA-SSD')
import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet.core.evaluation.kitti_eval import get_official_eval_result
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
import tools.kitti_common as kitti
import numpy as np
import torch.utils.data
import os
from tools.train_utils import load_params_from_file
from mmdet.datasets import utils

def single_test(model, data_loader, saveto=None, class_names=['Car']):
    template = '{} ' + ' '.join(['{:.4f}' for _ in range(15)]) + '\n'
    if saveto is not None:
        mmcv.mkdir_or_exist(saveto)

    model.eval()
    annos = []

    prog_bar = mmcv.ProgressBar(len(data_loader))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            results = model(return_loss=False, **data)
        annos+=results
        prog_bar.update()

    return annos


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    dataset = utils.get_dataset(cfg.data.val)
    class_names = cfg.data.val.class_names

    if args.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

        print("Evaluate on", cfg.data.val.class_names)
        setattr(model, 'class_names', class_names)

        #load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])
        load_params_from_file(model, args.checkpoint)
        data_loader = build_dataloader(
            dataset,
            1,
            cfg.data.workers_per_gpu,
            num_gpus=1,
            shuffle=False,
            dist=False)
        outputs = single_test(model, data_loader, args.out)
    else:
        NotImplementedError

    # kitti evaluation
    gt_annos = kitti.get_label_annos(dataset.label_prefix, dataset.sample_ids)
    result = get_official_eval_result(gt_annos, outputs, current_classes=class_names)
    print(result)


if __name__ == '__main__':
    main()
