import os
import json

import logging
logger = logging.getLogger(__name__)

import numpy as np
import math
import copy
import cv2
from tqdm import tqdm
from utils.vis import tabular_str, draw_skeleton

class Evaluator(object):
    def __init__(self, cfg, output_dir):
        self.output_dir = output_dir
        self.root = cfg.DATASET.ROOT
        self.dataset = cfg.DATASET.DATASET
        self.image_set = cfg.DATASET.TEST
        if self.dataset == 'crowdpose':
            from crowdposetools.coco import COCO
        else:
            from pycocotools.coco import COCO
        self.coco = COCO(os.path.join(self.root, 'annotations', '{}_{}.json'.format(self.dataset, self.image_set)))

    def evaluate(self, preds):
        # save json file
        res_folder = os.path.join(self.output_dir, "results")
        if not os.path.exists(res_folder): os.makedirs(res_folder)
        res_file = os.path.join(res_folder, "keypoints_%s_results.json" % (self.image_set))
        json.dump(preds, open(res_file, 'w'))

        # do eval with api
        if self.dataset == 'crowdpose':
            from crowdposetools.cocoeval import COCOeval
        else:
            from pycocotools.cocoeval import COCOeval
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        info_str = []
        if self.dataset == 'crowdpose':
            stats_names = ['AP', 'Ap .5', 'AP .75', 'APm', 'APl', 'AR', 'AR .5',
                           'AR .75', 'AP (easy)', 'AP (medium)', 'AP (hard)']
            stats_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for ind, name in enumerate(stats_names):
                info_str.append((name, coco_eval.stats[stats_index[ind]]))
        else:
            stats_names = ['AP', 'Ap .5', 'AP .75',
                           'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
            for ind, name in enumerate(stats_names):
                info_str.append((name, coco_eval.stats[ind]))

        return info_str
