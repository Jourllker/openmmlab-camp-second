# openmmlab-camp-second (OpenMMLab 实战营第二期)

## MMPose 和 MMDetection 作业 —— 耳朵穴位关键点检测

[数据集](https://pan.baidu.com/s/1swTLpArj7XEDXW4d0lo7Mg) ，提取码：741p \
数据集结构：

- 图片（"images"）：完全是图片，标注都在 coco 文件里面。
  coco.json：对图片的标号，对每张图片的目标检测框的标注("annotations/bbox")，对关键点的标注("annotations/keypoints")。

环境配置：[子豪兄](https://www.bilibili.com/video/BV12a4y1u7sd/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=b387f68d485f1edadd863fac222398ba) \
本次作业采用命令行形式进行，具体步骤见 ipynb 文件。

预测相关参数解释：
* bbox-thr kpt-thr nms-thr 这三个参数调大意味着被认为是关键点的“阈值”变大，门槛变高，更少的点被认为是关键点。这三个参数要反复自己调。我的是我自己调整过后的。
* radius 画在图上的点的半径，觉得画得太大了的话，可以调小一点。

## 作业总结

1、在关键点检测上出了问题，发现模型基本没有学到任何东西，不知道是哪里的问题。\
2、尝试将input_size、sigma、in_featuremap_size三个参数变大，以得到更好的模型性能。但是不管是x4还是x2，甚至将batch_size调到32，也会出现显存溢出的情况（RTX 3060 laptop 6G）。 \
3、重新再尝试训练一个关键点检测模型，现在的模型是正常的。
4、用新的模型进行预测评估和结果呈现。
