## 3D Human Pose Demo

<img  src="https://user-images.githubusercontent.com/15977946/118820606-02df2000-b8e9-11eb-9984-b9228101e780.gif"  width="600px"  alt><br>

### 3D Human Pose Two-stage Estimation Video Demo

#### Using mmdet for human bounding box detection and top-down model for the 1st stage (2D pose detection), and inference the 2nd stage (2D-to-3D lifting)

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

```shell
python  demo/body3d_pose_lifter_demo.py  \
${MMDET_CONFIG_FILE} \
${MMDET_CHECKPOINT_FILE} \
${MMPOSE_CONFIG_FILE_2D} \
${MMPOSE_CHECKPOINT_FILE_2D} \
${MMPOSE_CONFIG_FILE_3D} \
${MMPOSE_CHECKPOINT_FILE_3D} \
--input ${VIDEO_PATH} \
[--show] \
[--rebase-keypoint-height] \
[--norm-pose-2d] \
[--output-root ${OUT_VIDEO_ROOT}] \
[--device ${GPU_ID  or  CPU}] \
[--det-cat-id DET_CAT_ID] \
[--bbox-thr BBOX_THR] \
[--kpt-thr KPT_THR] \
[--use-oks-tracking] \
[--tracking-thr TRACKING_THR] \
[--thickness THICKNESS] \
[--radius RADIUS] \
[--use-multi-frames] [--online]
```

Note that

1. `${VIDEO_PATH}` can be the local path or **URL** link to video file.

2. You can turn on the `[--use-multi-frames]` option to use multi frames for inference in the 2D pose detection stage.

3. If the `[--online]` option is set to **True**, future frame information can **not** be used when using multi frames for inference in the 2D pose detection stage.

Examples:

During 2D pose detection, for single-frame inference that do not rely on extra frames to get the final results of the current frame, try this:

```shell
python  demo/body3d_pose_lifter_demo.py  \
demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth  \
configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth  \
configs/body_3d_keypoint/video_pose_lift/h36m/vid-pl_videopose3d-243frm-supv-cpn-ft_8xb128-200e_h36m.py \
https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth  \
--input https://user-images.githubusercontent.com/87690686/164970135-b14e424c-765a-4180-9bc8-fa8d6abc5510.mp4 \
--output-root  vis_results  \
--rebase-keypoint-height
```

During 2D pose detection, for multi-frame inference that rely on extra frames to get the final results of the current frame, try this:

```shell
python  demo/body3d_pose_lifter_demo.py  \
demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth  \
configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w48_8xb64-20e_posetrack18-384x288.py \
https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288-5fd6d3ff_20211130.pth  \
configs/body_3d_keypoint/video_pose_lift/h36m/vid-pl_videopose3d-243frm-supv-cpn-ft_8xb128-200e_h36m.py \
https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth  \
--input https://user-images.githubusercontent.com/87690686/164970135-b14e424c-765a-4180-9bc8-fa8d6abc5510.mp4 \
--output-root  vis_results  \
--rebase-keypoint-height \
--use-multi-frames  --online
```
