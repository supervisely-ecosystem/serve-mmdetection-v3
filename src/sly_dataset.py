from mmengine.dataset import BaseDataset, Compose
from mmdet.registry import DATASETS


@DATASETS.register_module()
class SuperviselyDatasetSplit(BaseDataset):
    def __init__(
        self,
        data_root: str,
        split_file: str,
        task: str,
        selected_classes: list = None,
        filter_images_without_gt: bool = True,
        save_coco_ann_file: str = None,
        serialize_data: bool = True,
        test_mode: bool = False,
        pipeline: list = [],
        max_refetch=1000,
        **kwargs
    ):
        # fake dataset for inference
        self._metainfo = {"classes": selected_classes, "palette": None}
