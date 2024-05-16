import os
import shutil
import pkg_resources
from collections import OrderedDict

from supervisely.app.widgets import (
    Widget,
    PretrainedModelsSelector,
    CustomModelsSelector,
    RadioTabs,
)

try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Any, Dict, Union
from pathlib import Path
import yaml
from dotenv import load_dotenv
import torch
import supervisely as sly
from supervisely.nn.prediction_dto import PredictionBBox, PredictionMask
from mmengine import Config
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from src.gui import MMDetectionGUI
from src import utils

# dataset registration (don't remove):
from src.sly_dataset import SuperviselyDatasetSplit
from supervisely.io.fs import get_file_name, silent_remove

root_source_path = str(Path(__file__).parents[1])
app_source_path = str(Path(__file__).parents[1])
load_dotenv(os.path.join(app_source_path, "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))

det_models_meta_path = os.path.join(root_source_path, "models", "detection_meta.json")
segm_models_meta_path = os.path.join(root_source_path, "models", "instance_segmentation_meta.json")

api = sly.Api.from_env()
team_id = sly.env.team_id()

use_gui_for_local_debug = bool(int(os.environ.get("USE_GUI", "1")))

configs_dir = os.path.join(root_source_path, "configs")
mmdet_ver = pkg_resources.get_distribution("mmdet").version
if os.path.isdir(f"/tmp/mmdet/mmdetection-{mmdet_ver}"):
    if os.path.isdir(configs_dir):
        shutil.rmtree(configs_dir)
    sly.logger.info(f"Getting model configs of current mmdetection version {mmdet_ver}...")
    shutil.copytree(f"/tmp/mmdet/mmdetection-{mmdet_ver}/configs", configs_dir)
    models_cnt = len(os.listdir(configs_dir)) - 1
    sly.logger.info(f"Found {models_cnt} models in {configs_dir} directory.")


class MMDetectionModel(sly.nn.inference.InstanceSegmentation):
    def initialize_custom_gui(self) -> Widget:
        """Create custom GUI layout for model selection. This method is called once when the application is started."""
        models = self.get_models()
        filtered_models = utils.filter_models_structure(models)
        self.pretrained_models_table = PretrainedModelsSelector(filtered_models)
        custom_models = sly.nn.checkpoints.mmdetection3.get_list(api, team_id)
        self.custom_models_table = CustomModelsSelector(
            team_id,
            custom_models,
            show_custom_checkpoint_path=True,
            custom_checkpoint_task_types=[
                "object detection",
                "instance segmentation",
            ],
        )

        self.model_source_tabs = RadioTabs(
            titles=["Pretrained models", "Custom models"],
            descriptions=["Publicly available models", "Models trained by you in Supervisely"],
            contents=[self.pretrained_models_table, self.custom_models_table],
        )
        return self.model_source_tabs

    def get_params_from_gui(self) -> dict:
        model_source = self.model_source_tabs.get_active_tab()
        self.device = self.gui.get_device()
        if model_source == "Pretrained models":
            model_params = self.pretrained_models_table.get_selected_model_params()
        elif model_source == "Custom models":
            model_params = self.custom_models_table.get_selected_model_params()
            if self.custom_models_table.use_custom_checkpoint_path():
                checkpoint_path = self.custom_models_table.get_custom_checkpoint_path()
                model_params["config_url"] = (
                    f"{os.path.dirname(checkpoint_path).rstrip('/')}/config.py"
                )
                file_info = api.file.exists(team_id, model_params["config_url"])
                if file_info is None:
                    raise FileNotFoundError(
                        f"Config file not found: {model_params['config_url']}. "
                        "Config should be placed in the same directory as the checkpoint file."
                    )

        self.selected_model_name = model_params.get("arch_type")
        self.checkpoint_name = model_params.get("checkpoint_name")
        self.task_type = model_params.get("task_type")

        deploy_params = {
            "device": self.device,
            **model_params,
        }
        return deploy_params

    def load_model_meta(
        self, model_source: str, cfg: Config, checkpoint_name: str = None, arch_type: str = None
    ):
        def set_common_meta(classes, task_type):
            obj_classes = [
                sly.ObjClass(
                    name, sly.Bitmap if task_type == "instance segmentation" else sly.Rectangle
                )
                for name in classes
            ]
            self.selected_model_name = arch_type
            self.checkpoint_name = checkpoint_name
            self.dataset_name = cfg.dataset_type
            self.class_names = classes
            self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
            self._get_confidence_tag_meta()

        if model_source == "Custom models":
            is_custom_checkpoint_path = self.custom_models_table.use_custom_checkpoint_path()
            if is_custom_checkpoint_path and cfg.dataset_type != "SuperviselyDatasetSplit":

                # classes from .pth
                classes = self.model.dataset_meta.get("classes", [])
                if classes == []:
                    # classes from config
                    dataset_class_name = cfg.dataset_type
                    dataset_meta = DATASETS.module_dict[dataset_class_name].METAINFO
                    classes = dataset_meta.get("classes", [])
                    if classes == []:
                        raise ValueError("Classes not found in the .pth and config file")
                self.dataset_name = cfg.dataset_type
                set_common_meta(classes, self.task_type)

            else:
                classes = cfg.train_dataloader.dataset.selected_classes
                self.selected_model_name = cfg.sly_metadata.architecture_name
                self.checkpoint_name = checkpoint_name
                self.dataset_name = cfg.sly_metadata.project_name
                self.task_type = cfg.sly_metadata.task_type.replace("_", " ")
                set_common_meta(classes, self.task_type)

        elif model_source == "Pretrained models":
            dataset_class_name = cfg.dataset_type
            dataset_meta = DATASETS.module_dict[dataset_class_name].METAINFO
            classes = dataset_meta["classes"]
            self.dataset_name = cfg.dataset_type
            set_common_meta(classes, self.task_type)

        self.model.test_cfg["score_thr"] = 0.45  # default confidence_thresh

    def load_model(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        model_source: Literal["Pretrained models", "Custom models"],
        task_type: Literal["object detection", "instance segmentation"],
        checkpoint_name: str,
        checkpoint_url: str,
        config_url: str,
        arch_type: str = None,
    ):
        """
        Load model method is used to deploy model.

        :param model_source: Specifies whether the model is pretrained or custom.
        :type model_source: Literal["Pretrained models", "Custom models"]
        :param device: The device on which the model will be deployed.
        :type device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        :param task_type: The type of task the model is designed for.
        :type task_type: Literal["object detection", "instance segmentation"]
        :param checkpoint_name: The name of the checkpoint from which the model is loaded.
        :type checkpoint_name: str
        :param checkpoint_url: The URL where the model checkpoint can be downloaded.
        :type checkpoint_url: str
        :param config_url: The URL where the model config can be downloaded.
        :type config_url: str
        :param arch_type: The architecture type of the model.
        :type arch_type: str
        """
        self.device = device
        self.task_type = task_type

        local_weights_path = os.path.join(self.model_dir, checkpoint_name)
        if model_source == "Pretrained models":
            if not sly.fs.file_exists(local_weights_path):
                self.download(
                    src_path=checkpoint_url,
                    dst_path=local_weights_path,
                )
            local_config_path = os.path.join(root_source_path, config_url)
        else:
            self.download(
                src_path=checkpoint_url,
                dst_path=local_weights_path,
            )
            local_config_path = os.path.join(configs_dir, "custom", "config.py")
            if sly.fs.file_exists(local_config_path):
                silent_remove(local_config_path)
            self.download(
                src_path=config_url,
                dst_path=local_config_path,
            )
            if not sly.fs.file_exists(local_config_path):
                raise FileNotFoundError(
                    f"Config file not found: {config_url}. "
                    "Config should be placed in the same directory as the checkpoint file."
                )

        try:
            cfg = Config.fromfile(local_config_path)
            if "pretrained" in cfg.model:
                cfg.model.pretrained = None
            elif "init_cfg" in cfg.model.backbone:
                cfg.model.backbone.init_cfg = None
            cfg.model.train_cfg = None
            self.model = init_detector(
                cfg, checkpoint=local_weights_path, device=device, palette=[]
            )
            self.load_model_meta(model_source, cfg, checkpoint_name, arch_type)
        except KeyError as e:
            raise KeyError(f"Error loading config file: {local_config_path}. Error: {e}")

    def get_info(self) -> dict:
        info = super().get_info()
        info["model_name"] = self.selected_model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        info["task type"] = self.task_type
        info["videos_support"] = True
        info["async_video_inference_support"] = True
        info["tracking_on_videos_support"] = True
        return info

    def get_models(self):
        tasks = ["object detection", "instance segmentation"]
        model_config = {}
        for task_type in tasks:
            model_config[task_type] = {}
            if task_type == "object detection":
                models_meta_path = det_models_meta_path
            elif task_type == "instance segmentation":
                models_meta_path = segm_models_meta_path
            model_yamls = sly.json.load_json_file(models_meta_path)

            for model_meta in model_yamls:
                mmdet_ver = pkg_resources.get_distribution("mmdet").version
                model_yml_url = f"https://github.com/open-mmlab/mmdetection/tree/v{mmdet_ver}/configs/{model_meta['yml_file']}"
                model_yml_local = os.path.join(configs_dir, model_meta["yml_file"])
                with open(model_yml_local, "r") as stream:
                    model_info = yaml.safe_load(stream)
                    model_config[task_type][model_meta["model_name"]] = {}
                    model_config[task_type][model_meta["model_name"]]["checkpoints"] = []
                    model_config[task_type][model_meta["model_name"]]["paper_from"] = model_meta[
                        "paper_from"
                    ]
                    model_config[task_type][model_meta["model_name"]]["year"] = model_meta["year"]
                    model_config[task_type][model_meta["model_name"]]["config_url"] = (
                        os.path.dirname(model_yml_url)
                    )

                    models = model_info if isinstance(model_info, list) else model_info["Models"]
                    for model in models:
                        checkpoint_info = OrderedDict()

                        # exclude checkpoints
                        if utils.is_exclude(model["Name"], model_meta.get("exclude")):
                            continue

                        checkpoint_info["Model"] = model["Name"]
                        checkpoint_info["Method"] = model["In Collection"]
                        if "Weights" not in model.keys():
                            # skip models without weights
                            continue
                        try:
                            checkpoint_info["Inference Time (ms/im)"] = model["Metadata"][
                                "inference time (ms/im)"
                            ][0]["value"]
                        except KeyError:
                            checkpoint_info["Inference Time (ms/im)"] = "-"
                        try:
                            checkpoint_info["Input Size (H, W)"] = model["Metadata"][
                                "inference time (ms/im)"
                            ][0]["resolution"]
                        except KeyError:
                            checkpoint_info["Input Size (H, W)"] = "-"
                        try:
                            checkpoint_info["LR scheduler (epochs)"] = model["Metadata"]["Epochs"]
                        except KeyError:
                            checkpoint_info["LR scheduler (epochs)"] = "-"
                        try:
                            checkpoint_info["Memory (Training, GB)"] = model["Metadata"][
                                "Training Memory (GB)"
                            ]
                        except KeyError:
                            checkpoint_info["Memory (Training, GB)"] = "-"
                        for result in model["Results"]:
                            if (
                                task_type == "object detection"
                                and result["Task"] == "Object Detection"
                            ) or (
                                task_type == "instance segmentation"
                                and result["Task"] == "Instance Segmentation"
                            ):
                                checkpoint_info["Dataset"] = result["Dataset"]
                                for metric_name, metric_val in result["Metrics"].items():
                                    checkpoint_info[metric_name] = metric_val
                        try:
                            weights_file = model["Weights"]
                        except KeyError as e:
                            sly.logger.info(
                                f'Weights not found. Model: {model_meta["model_name"]}, checkpoint: {checkpoint_info["Model"]}'
                            )
                            continue
                        checkpoint_info["meta"] = {
                            "task_type": None,
                            "arch_type": None,
                            "arch_link": None,
                            "weights_url": model["Weights"],
                            "config_url": model["Config"],
                        }

                        model_config[task_type][model_meta["model_name"]]["checkpoints"].append(
                            checkpoint_info
                        )
        return model_config

    def get_classes(self) -> List[str]:
        return self.class_names

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[Union[PredictionBBox, PredictionMask]]:
        # set confidence_thresh
        conf_tresh = settings.get("confidence_thresh", 0.45)
        if conf_tresh:
            # TODO: may be set recursively?
            self.model.test_cfg["score_thr"] = conf_tresh

        # set nms_iou_thresh
        nms_tresh = settings.get("nms_iou_thresh", 0.65)
        if nms_tresh:
            test_cfg = self.model.test_cfg
            if hasattr(test_cfg, "nms"):
                test_cfg["nms"]["iou_threshold"] = nms_tresh
            if hasattr(test_cfg, "rcnn") and hasattr(test_cfg["rcnn"], "nms"):
                test_cfg["rcnn"]["nms"]["iou_threshold"] = nms_tresh
            if hasattr(test_cfg, "rpn") and hasattr(test_cfg["rpn"], "nms"):
                test_cfg["rpn"]["nms"]["iou_threshold"] = nms_tresh

        # inference
        result: DetDataSample = inference_detector(self.model, image_path)
        preds = result.pred_instances.cpu().numpy()

        # collect predictions
        predictions = []
        for pred in preds:
            pred: InstanceData
            score = float(pred.scores[0])
            if conf_tresh is not None and score < conf_tresh:
                # filter by confidence
                continue
            class_name = self.class_names[pred.labels.astype(int)[0]]
            if self.task_type == "object detection":
                x1, y1, x2, y2 = pred.bboxes[0].astype(int).tolist()
                tlbr = [y1, x1, y2, x2]
                sly_pred = PredictionBBox(class_name=class_name, bbox_tlbr=tlbr, score=score)
            else:
                if pred.get("masks") is None:
                    raise Exception(
                        f'The model "{self.checkpoint_name}" can\'t predict masks. Please, try another model.'
                    )
                mask = pred.masks[0]
                sly_pred = PredictionMask(class_name=class_name, mask=mask, score=score)
            predictions.append(sly_pred)

        # TODO: debug
        # ann = self._predictions_to_annotation(image_path, predictions)
        # img = sly.image.read(image_path)
        # ann.draw_pretty(img, thickness=2, opacity=0.4, output_path="test.jpg")
        return predictions


if sly.is_production():
    sly.logger.info(
        "Script arguments",
        extra={
            "context.teamId": sly.env.team_id(),
            "context.workspaceId": sly.env.workspace_id(),
        },
    )

custom_settings_path = os.path.join(app_source_path, "custom_settings.yml")

m = MMDetectionModel(use_gui=True, custom_inference_settings=custom_settings_path)

if sly.is_production() or use_gui_for_local_debug is True:
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging without GUI
    # task_type = "object detection"
    # models = utils.get_models()[task_type]
    # selected_model_name = "TOOD"
    # dataset_name = "COCO"
    # selected_checkpoint = models[selected_model_name]["checkpoints"][0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    deploy_params = m.get_params_from_gui()
    m.load_model(**deploy_params)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, m.custom_inference_settings_dict)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
