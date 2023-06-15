from mmengine.dataset import BaseDataset, Compose
from mmdet.registry import DATASETS
from typing import List
import numpy as np
import supervisely as sly
from supervisely.project.project import ItemInfo
import pycocotools.mask


@DATASETS.register_module()
class SuperviselyDatasetSplit(BaseDataset):
    def __init__(
        self,
        data_root: str,
        split_file: str,
        selected_classes: list = None,
        filter_images_without_gt: bool = False,
        save_coco_ann_file: str = None,
        serialize_data: bool = True,
        test_mode: bool = False,
        pipeline: list = [],
        max_refetch=1000,
    ):
        # fake dataset for inference
        pass
