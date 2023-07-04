<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/serve-mmdetection-2/assets/115161827/b5d59262-637a-491c-97da-5984ffac6319"/>  

# Serve MMDetection 3.0
  
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

Serve MMDetection model as a Supervisely Application. MMDetection is an open-source toolbox based on PyTorch. Learn more about MMDetection and available models [here](https://github.com/open-mmlab/mmdetection).

MMDetection 3.0.0 is being released as the official version, achieving new state-of-the-art performance in instance segmentation and rotated object detection tasks.

Application key points:
- All Object Detection and Instance Segmentation models from MM Toolbox are available
- Deployed on GPU or CPU

# How to Run

0. Start the application from an app's context menu or the Ecosystem.

1. Select a Deep Learning problem to solve.

<img src="https://github.com/supervisely-ecosystem/serve-mmdetection-v3/assets/115161827/099a91f6-1c78-4d73-a1e8-c49b60065a09" />

**Pretrained models**

2. Select architecture, pre-trained model, and deploying device and press the `Serve` button.

<img src="https://github.com/supervisely-ecosystem/serve-mmdetection-v3/assets/115161827/7a2d820a-12ef-4230-9630-90c3a67bad17" />

**Custom models**

Model and directory structure must be acquired via [Train MMDetection V3](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/train-mmdetection-v3) app or manually created with the same directory structure.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/train-mmdetection-v3" src="https://github.com/supervisely-ecosystem/serve-mmdetection-v3/assets/115161827/9a995648-3d4a-48a3-b89b-f395ae335086" height="60px" margin-bottom="20px"/>

3. Wait for the model to deploy.

<img src="https://github.com/supervisely-ecosystem/serve-mmdetection-v3/assets/115161827/9e0afbc0-a8bc-4892-9033-5ebfda88ad5e" />


# How To Use Custom Model Outside The Platform

We attach [the notebook](https://github.com/supervisely-ecosystem/mmdetection/blob/main/serve/run_image.ipynb) as a template to use your custom model outside Supervisely. Your model should be trained in [Train MMDetection](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmdetection/train) app and the checkpoint should be saved to Team Files.

1. Download weights file `.pth` from Team Files. Select a file with the name `best_*.pth` to use a model with the best score or any other model from the `checkpoints` directory.
2. Download the `config.py` file from Team Files (`checkpoints` directory).
3. Change paths in the notebook (variables `img_path`, `config_path`, `weights_path`, `device`).
4. Run the notebook step by step.

# Related Apps


- [Train MMDetection V3](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/train-mmdetection-v3) - app allows to play with different inference options, monitor metrics charts in real time, and save training artifacts to Team Files.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/train-mmdetection-v3" src="https://github.com/supervisely-ecosystem/serve-mmdetection-v3/assets/115161827/9a995648-3d4a-48a3-b89b-f395ae335086" height="70px" margin-bottom="20px"/>

- [Apply NN to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyze predictions and perform automatic data pre-labeling.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" height="70px" margin-bottom="20px"/>  

- [Apply NN to Videos Project](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press the `Apply` button (or use a hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>

# Acknowledgment

This app is based on the great work `MMDetection` ([github](https://github.com/open-mmlab/mmdetection)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmdetection?style=social)


