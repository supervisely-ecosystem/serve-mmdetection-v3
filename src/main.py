import os
import shutil
import sys
import pkg_resources
from collections import OrderedDict

try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Any, Dict, Union
from pathlib import Path
import numpy as np
import yaml
from dotenv import load_dotenv
import torch
import supervisely as sly
import supervisely.app.widgets as Widgets
import supervisely.nn.inference.gui as GUI
from supervisely.nn.prediction_dto import PredictionBBox, PredictionMask
from mmengine import Config
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData, PixelData
from src.sly_dataset import SuperviselyDatasetSplit  # don't remove, needed for dataset registration
from src.gui import MMDetectionGUI

root_source_path = str(Path(__file__).parents[1])
app_source_path = str(Path(__file__).parents[1])
load_dotenv(os.path.join(app_source_path, "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))

use_gui_for_local_debug = bool(int(os.environ.get("USE_GUI", "1")))

# models_meta_path = os.path.join(root_source_path, "models", "detection_meta.json")


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
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ) -> None:
        self.device = device
        if self.gui is not None:
            self.task_type = self.gui.get_task_type()
            model_source = self.gui.get_model_source()
            if model_source == "Pretrained models":
                selected_model = self.gui.get_checkpoint_info()
                weights_path, config_path = self.download_pretrained_files(
                    selected_model, model_dir
                )
            elif model_source == "Custom models":
                custom_weights_link = self.gui.get_custom_link()
                weights_path, config_path = self.download_custom_files(
                    custom_weights_link, model_dir
                )
        else:
            # for local debug without GUI only
            self.task_type = task_type
            model_source = "Pretrained models"
            weights_path, config_path = self.download_pretrained_files(
                selected_checkpoint, model_dir
            )
        cfg = Config.fromfile(config_path)
        if "pretrained" in cfg.model:
            cfg.model.pretrained = None
        elif "init_cfg" in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
        cfg.model.train_cfg = None
        model = init_detector(cfg, checkpoint=weights_path, device=device)

        if model_source == "Custom models":
            classes = cfg.checkpoint_config.meta.CLASSES
            self.selected_model_name = cfg.pretrained_model
            self.checkpoint_name = "custom"
            self.dataset_name = "custom"
            if "segm" in cfg.evaluation.metric:
                obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]
            else:
                obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in classes]
        elif model_source == "Pretrained models":
            dataset_class_name = cfg.dataset_type
            dataset_meta = DATASETS.module_dict[dataset_class_name].METAINFO
            classes = dataset_meta["classes"]
            if self.task_type == "object detection":
                obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in classes]
            elif self.task_type == "instance segmentation":
                obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]
            if self.gui is not None:
                self.selected_model_name = list(self.gui.get_model_info().keys())[0]
                checkpoint_info = self.gui.get_checkpoint_info()
                self.checkpoint_name = checkpoint_info["Name"]
                self.dataset_name = checkpoint_info["Dataset"]
            else:
                self.selected_model_name = selected_model_name
                self.checkpoint_name = selected_checkpoint["Name"]
                self.dataset_name = dataset_name

        self.model = model
        self.model.test_cfg["score_thr"] = 0.5  # default confidence_thresh
        self.class_names = classes
        sly.logger.debug(f"classes={classes}")

        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

        # TODO: remove test
        self.predict("demo_data/image_01.jpg", {})

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = self.task_type
        info["model_name"] = self.selected_model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_models(self, add_links=False):
        tasks = ["object detection", "instance segmentation"]
        model_config = {}
        for task_type in tasks:
            model_config[task_type] = {}
            if task_type == "object detection":
                # models_meta_path = os.path.join(root_source_path, "models", "detection_meta.json")
                models_meta_path = os.path.join(root_source_path, "models", "det_models.json")
            elif task_type == "instance segmentation":
                models_meta_path = os.path.join(
                    # root_source_path, "models", "instance_segmentation_meta.json"
                    root_source_path,
                    "models",
                    "segm_models.json",
                )
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
                    model_config[task_type][model_meta["model_name"]][
                        "config_url"
                    ] = os.path.dirname(model_yml_url)

                    models = model_info if isinstance(model_info, list) else model_info["Models"]
                    for model in models:
                        checkpoint_info = OrderedDict()
                        if "exclude" in model_meta.keys():
                            if model_meta["exclude"].endswith("*"):
                                if model["Name"].startswith(model_meta["exclude"][:-1]):
                                    continue
                        # Saved For Training
                        # checkpoint_info["Use semantic inside"] = False
                        # if "semantic" in model_meta.keys():
                        #     if model_meta["semantic"] == "*":
                        #         checkpoint_info["Use semantic inside"] = True
                        #     elif model_meta["semantic"].startswith("*") and model_meta["semantic"].endswith("*"):
                        #         if model_meta["semantic"][1:-1] in model["Name"]:
                        #             checkpoint_info["Use semantic inside"] = True
                        #     elif model_meta["semantic"].startswith("*") and model["Name"].endswith(model_meta["semantic"][1:]):
                        #         checkpoint_info["Use semantic inside"] = True
                        #     elif model_meta["semantic"].endswith("*") and model_meta["semantic"].startswith("!"):
                        #         if not model["Name"].startswith(model_meta["semantic"][1:-1]):
                        #             checkpoint_info["Use semantic inside"] = True

                        checkpoint_info["Name"] = model["Name"]
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
                                f'Weights not found. Model: {model_meta["model_name"]}, checkpoint: {checkpoint_info["Name"]}'
                            )
                            continue
                        if add_links:
                            checkpoint_info["config_file"] = model["Config"]
                            checkpoint_info["weights_file"] = weights_file
                        model_config[task_type][model_meta["model_name"]]["checkpoints"].append(
                            checkpoint_info
                        )
        return model_config

    def download_pretrained_files(self, selected_model: Dict[str, str], model_dir: str):
        gui: MMDetectionGUI
        task_type = self.gui.get_task_type()
        models = self.get_models(add_links=True)[task_type]
        if self.gui is not None:
            model_name = list(self.gui.get_model_info().keys())[0]
        else:
            # for local debug without GUI only
            model_name = selected_model_name
        full_model_info = selected_model
        for model_info in models[model_name]["checkpoints"]:
            if model_info["Name"] == selected_model["Name"]:
                full_model_info = model_info
        weights_ext = sly.fs.get_file_ext(full_model_info["weights_file"])
        config_ext = sly.fs.get_file_ext(full_model_info["config_file"])
        weights_dst_path = os.path.join(model_dir, f"{selected_model['Name']}{weights_ext}")
        if not sly.fs.file_exists(weights_dst_path):
            self.download(src_path=full_model_info["weights_file"], dst_path=weights_dst_path)
        config_path = self.download(
            src_path=full_model_info["config_file"],
            dst_path=os.path.join(model_dir, f"config{config_ext}"),
        )

        return weights_dst_path, config_path

    def download_custom_files(self, custom_link: str, model_dir: str):
        weight_filename = os.path.basename(custom_link)
        weights_dst_path = os.path.join(model_dir, weight_filename)
        if not sly.fs.file_exists(weights_dst_path):
            self.download(
                src_path=custom_link,
                dst_path=weights_dst_path,
            )
        config_path = self.download(
            src_path=os.path.join(os.path.dirname(custom_link), "config.py"),
            dst_path=os.path.join(model_dir, "config.py"),
        )

        return weights_dst_path, config_path

    def initialize_gui(self) -> None:
        models = self.get_models()
        for task_type in ["object detection", "instance segmentation"]:
            for model_group in models[task_type].keys():
                models[task_type][model_group]["checkpoints"] = self._preprocess_models_list(
                    models[task_type][model_group]["checkpoints"]
                )
        self._gui = MMDetectionGUI(
            models,
            self.api,
            support_pretrained_models=True,
            support_custom_models=True,
            custom_model_link_type="file",
        )

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[Union[PredictionBBox, PredictionMask]]:
        # set confidence_thresh
        conf_tresh = settings.get("confidence_thresh", 0.5)
        if conf_tresh:
            self.model.test_cfg["score_thr"] = conf_tresh

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
            class_name = self.class_names[pred.labels[0]]
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

        # TODO: test
        ann = self._predictions_to_annotation(image_path, predictions)
        img = sly.image.read(image_path)
        ann.draw_pretty(img, thickness=2, opacity=0.4, output_path="test.jpg")
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
    task_type = "object detection"
    models = m.get_models(add_links=True)[task_type]
    selected_model_name = "TOOD"
    dataset_name = "COCO"
    selected_checkpoint = models[selected_model_name]["checkpoints"][0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, m.custom_inference_settings_dict)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
