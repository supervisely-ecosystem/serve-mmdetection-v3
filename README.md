<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/serve-mmdetection-2/assets/115161827/b5d59262-637a-491c-97da-5984ffac6319"/>  

# Serve MMDetection v3
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use-Custom-Model-Outside-The-Platform">How To Use Custom Model Outside The Platform</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-mmdetection-v3)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-mmdetection-v3)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/serve-mmdetection-v3.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/serve-mmdetection-v3.png)](https://supervise.ly)
 
</div>

# Overview

Serve MMDetection model as a Supervisely Application. MMDetection is an open source toolbox based on PyTorch. Learn more about MMDetection and available models [here](https://github.com/open-mmlab/mmdetection).

:star2: MMDetection 3.0.0 is being released as the official version, having achieved new state-of-the-art performance in instance segmentation and rotated object detection tasks.

Application key points:
- All Object Detection and Instance Segmentation models from MM Toolbox are available
- Deployed on GPU or CPU

# How to Run

0. Start the application from the context menu of an app or Ecosystem.

**Pretrained models**

1. Select Deep Learning problem to solve.

<img src="XXX" />

2. Select architecture, pretrained model and deploying device and press the `Serve` button.

<img src="XXX" />

3. Wait for the model to deploy.

  <img src="XXX" />

**Custom models**

Model and directory structure must be acquired via Train MMDetection app or manually created with the same directory structure.

<img src="XXX" />

# How To Use Custom Model Outside The Platform

We attach [the notebook](https://github.com/supervisely-ecosystem/mmdetection/blob/main/serve/run_image.ipynb) as template to use your custom model outside Supervisely. Your model should be trained in [Train MMDetection](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmdetection/train) app and checkpoint should be saved to Team Files.

1. Download weights file `.pth` from Team Files. Select file with name `best_*.pth` to use model with the best score or any other model from `checkpoints` directory.
2. Download `config.py` file from Team Files (`checkpoints` directory).
3. Change paths in the notebook (variables `img_path`, `config_path`, `weights_path`, `device`).
4. Run the notebook step by step.

# Related Apps


- [Train MMDetection V3](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/train-mmdetection-2) - app allows to play with different inference options, monitor metrics charts in real time, and save training artifacts to Team Files.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/train-mmdetection-2" src="XXX" height="60px" margin-bottom="20px"/>

- [Apply NN to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" height="70px" margin-bottom="20px"/>  

- [Apply NN to Videos Project](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>

# Acknowledgment

This app is based on the great work `MMDetection` ([github](https://github.com/open-mmlab/mmdetection)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmdetection?style=social)


