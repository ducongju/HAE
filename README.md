# Hierarchical Associative Encoding and Decoding for Bottom-Up Human Pose Estimation

## Introduction
Bottom-up human pose estimation decouples computational complexity from the number of people but requires additional operations to match the detected keypoints to each human instance. Existing approaches treat all keypoints equally while ignoring the relationships among keypoints, which in turn limits the performance ceilings. In this work, we propose a hierarchical associative encoding and decoding framework for bottom-up human pose estimation by introducing additional prior knowledge. Specifically, in addition to keypoint-level and instance-level associations, we further divide keypoints into groups and pursue group-level associations. In this way, prior knowledge is incorporated to determine the keypoint groups for better associative encoding. To deal with complex poses, a focal pulling loss is proposed to focus more on the hard-to-associate keypoints. Moreover, instead of using a pre-defined order for keypoint grouping, we propose a progressive associative decoding method to dynamically determine the order of keypoints for grouping, which helps reduce isolated keypoints. Experimental results on both MS-COCO and CrowdPose datasets demonstrate the superior performance of the proposed associative encoding and decoding algorithms. More importantly, we prove, through validation, that hierarchical associative encoding and decoding can be used as a plug-n-play module for performance improvement regardless of backbone architectures.

## Main Results
### Results on COCO val2017
| Backbone | Input size | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|--------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** | 512x512 | 64.6 | 87.9 |  71.4 |  58.1 |  73.9 |  71.5 |  91.6 |  77.2 |  63.0 |  82.8 |
| **pose_higher_hrnet_w32** | 512x512 | 67.8 |  89.2 |  74.8 |  62.6 |  75.4 |  73.6 |  92.1 |  79.4 |  66.6 |  83.0 |

### Results on COCO test-dev2017
| Backbone | Input size | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|--------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** | 512x512 | 64.9 | 86.5 |  70.5 |  55.5 |  69.5 |  71.6 |  90.4 |  76.2 |  62.8 |  83.7 |
| **pose_higher_hrnet_w32** | 512x512 | 68.8 |  88.4 |  74.6 |  62.2 |  78.1 |  74.3 |  91.6 |  79.1 |  67.4 |  83.8 |

### Results on CrowdPose test
| Backbone             |    AP | Ap .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_higher_hrnet_w32** |67.6 | 87.9 |  73.2 |  75.4 |  68.3 |  59.9 |


## Environment

The code is developed using python 3.7, torch 1.10, torchvision 0.11 on cuda 11.1. NVIDIA GPUs are needed. The code is developed and tested using 2 NVIDIA A100 GPU cards for HrHRet-W32. Other platforms are not fully tested.

## Quick start

### Prepare the directory

1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.

2. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir work_dirs
   mkdir test_dirs
   mkdir test_results
   mkdir vis_input
   mkdir vis_output
   mkdir data
   ```

### Data preparation

**For [COCO](http://cocodataset.org/) data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing). Optionally, to evaluate on COCO'2017 test-dev, please download the [image-info](https://download.openmmlab.com/mmpose/datasets/person_keypoints_test-dev-2017.json). Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- |-- coco
        `-- │-- annotations
                │   │-- person_keypoints_train2017.json
                │   |-- person_keypoints_val2017.json
                │   |-- person_keypoints_test-dev-2017.json
                |-- person_detection_results
                |   |-- COCO_val2017_detections_AP_H_56_person.json
                |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
                │-- train2017
                │   │-- 000000000009.jpg
                │   │-- 000000000025.jpg
                │   │-- 000000000030.jpg
                │   │-- ...
                `-- val2017
                    │-- 000000000139.jpg
                    │-- 000000000285.jpg
                    │-- 000000000632.jpg
                    │-- ...

**For [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) data**, please download from [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose). Please download the annotation files and human detection results from [crowdpose_annotations](https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar). Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- │-- crowdpose
        `-- │-- annotations
            │   │-- mmpose_crowdpose_train.json
            │   │-- mmpose_crowdpose_val.json
            │   │-- mmpose_crowdpose_trainval.json
            │   │-- mmpose_crowdpose_test.json
            │   │-- det_for_crowd_test_0.1_0.5.json
            `--  images
                │-- 100000.jpg
                │-- 100001.jpg
                │-- 100002.jpg
                │-- ...

**For [MPII](http://human-pose.mpi-inf.mpg.de/) data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). We have converted the original annotation files into json format, please download them from [mpii_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar). Extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- │-- mpii
        `-- |-- annotations
            |   |-- mpii_gt_val.mat
            |   |-- mpii_test.json
            |   |-- mpii_train.json
            |   |-- mpii_trainval.json
            |   `-- mpii_val.json
            `-- images
                |-- 000001163.jpg
                |-- 000003072.jpg
                │-- ...


### Download the pretrained models

Download pretrained models and our well-trained models from zoo ([Google Drive](https://drive.google.com/drive/folders/1elE8kvlB7ctyslSmjhBGVV2eMPX3-rTN?usp=sharing) and make models directory look like this:

    ${POSE_ROOT}
    |-- work_dir       
    `-- |-- |-- HAE_COCO.pth
            `-- HAE_CrowdPose.pth

### Prepare the environment

If you are using SLURM (Simple Linux Utility for Resource Management), then execute:

```
sbatch prepare.sh
```

If you like, you can prepare the environment [**step by step**](https://github.com/open-mmlab/mmpose).

### Training and Testing

#### Testing on COCO val2017 dataset using well-trained pose model

```
./tools/dist_test.sh configs/HAE_COCO.py work_dirs/HAE_COCO.pth 1 --out test_results/HAE_COCO.json --eval mAP
```

#### Testing on CrowdPose test dataset using well-trained pose model

```
./tools/dist_test.sh configs/HAE_CrowdPose.py work_dirs/HAE_CrowdPose.pth 1 --out test_results/HAE_CrowdPose.json --eval mAP
```

#### Testing on MPII dataset using well-trained pose model

```
./tools/dist_test.sh configs/HAE_MPII.py work_dirs/HAE_MPII.pth 1 --out test_results/HAE_MPII.json --eval PCKh
```

#### Training on COCO train2017 dataset

```
./tools/dist_train.sh configs/HAE_COCO.py 2 --work-dir work_dirs/HAE_COCO --cfg-options evaluation.interval=100 model.pretrained=checkpoints/hrnet_w32-36af842e.pth
```

#### Training on Crowdpose trainval dataset

```
./tools/dist_train.sh configs/HAE_CrowdPose.py 2 --work-dir work_dirs/HAE_CrowdPose --cfg-options evaluation.interval=100 model.pretrained=checkpoints/hrnet_w32-36af842e.pth
```

#### Training on MPII dataset

```
./tools/dist_train.sh configs/HAE_MPII.py 2 --work-dir work_dirs/HAE_MPII --cfg-options evaluation.interval=100 model.pretrained=checkpoints/hrnet_w32-36af842e.pth
```

#### Using heatmap and tagmap demo

```
python demo/bottom_up_img_demo.py configs/higherhrnet_w32_crowdpose_512x512.py work_dirs/HAE_CrowdPose.pth --img-path vis_input/ --out-img-root vis_output --thickness 2 --radius 5 --kpt-thr 0.3 --pose-nms-thr 0.9
```

The above command will create three images under *vis_output* directory including the predicted keypoint coordinates, the predicted heatmap, and the predicted tagmap. 

### Acknowledge
Thanks for the open-source [MMPose](https://github.com/open-mmlab/mmpose), it is a part of the [OpenMMLab](https://github.com/open-mmlab/) project.
### Other implementations
* [HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). It is a part of the [HRNet](https://github.com/HRNet) project.
