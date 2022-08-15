# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Congju Du (ducongju@hust.edu.cn).

import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import mmcv
import cv2
import numpy as np
from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--img-path',
        type=str,
        help='Path to an image file or a image folder.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()
    assert args.show or (args.out_img_root != '')

    # prepare image list
    if osp.isfile(args.img_path):
        image_list = [args.img_path]
    elif osp.isdir(args.img_path):
        image_list = [
            osp.join(args.img_path, fn) for fn in os.listdir(args.img_path)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
    else:
        raise ValueError('Image path should be an image or image folder.'
                         f'Got invalid image path: {args.img_path}')

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = True
    return_tagmap = True

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # process each image
    for image_name in mmcv.track_iter_progress(image_list):

        # test a single image, with a list of bboxes.
        # mmpose/apis/inference.py
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            return_tagmap=return_tagmap,
            outputs=output_layer_names)

        tagmap_show = vis_tagmap(image_name, returned_outputs[0]['tagmap'][0][:,:,:,0])
        heatmap_show = vis_heatmap(image_name, returned_outputs[0]['heatmap'][0])

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(
                args.out_img_root,
                f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')
        
        # show the tagmap
        out_file_tagmap = os.path.join(
            args.out_img_root, 'tagmap',
            f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')
        cv2.imwrite(out_file_tagmap, tagmap_show)

        # show the heatmap
        out_file_heatmap = os.path.join(
            args.out_img_root, 'heatmap',
            f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')
        cv2.imwrite(out_file_heatmap, heatmap_show)

        # show the results
        # mmpose/apis/inference.py
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_file)

def vis_tagmap(image_name, tagmaps):
    image = cv2.imread(image_name)
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))
    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = ((tagmap - min) * 255 / (max - min + 1e-5)).clip(0, 255).astype(np.uint8)
        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap*0.9 + image_resized*0.1
        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized
    return image_grid

def vis_heatmap(image_name, heatmaps):
    image = cv2.imread(image_name)
    heatmaps = (heatmaps * 255).clip(0, 255).astype(np.uint8)
    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))
    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized
    return image_grid

if __name__ == '__main__':
    main()
