

![](img/cvprocesslib.png)

# CV-Process-Lib

---

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/isLinXu/CVProcessLib) ![img](https://badgen.net/badge/icon/vison?icon=awesome&label)
![](https://badgen.net/github/stars/isLinXu/CVProcessLib)![](https://badgen.net/github/forks/isLinXu/CVProcessLib)![](https://badgen.net/github/prs/isLinXu/CVProcessLib)![](https://badgen.net/github/releases/isLinXu/CVProcessLib)![](https://badgen.net/github/license/isLinXu/CVProcessLib)![img](https://hits.dwyl.com/isLinXu/CVProcessLib.svg)

## 一、介绍

本项目为个人基于Opecv、numpy等Python开发包所建立维护的图像处理与开发库。

## 二、效果展示







## 三、用法







## 四、处理库架构

### 4.1 基本介绍

该项目主要以`Python`为主要开发语言进行编写。

按照金字塔分层设计的思想，分为以下三层：

底层设计为独立模块**[core]**，主要为一些通用性、功能性的核心组件

中层设计为业务封装**[package]**，主要根据项目业务需求调用不同的组件和模型进行封装使用；

顶层设计为独立项目**[project]**，主要与公司产品业务需求进行对接。

### 4.2 目录树结构

```shell
CVProcessLib/
├── core
│   ├── board.py
│   ├── cv
│   │   ├── arucoMarkerCode
│   │   │   ├── aruco_dection.py
│   │   │   ├── aruco_dect.py
│   │   │   ├── arucoMaker.py
│   │   │   └── dlq4b02.py
│   │   ├── basic_process
│   │   │   └── imageprocess.py
│   │   ├── camlibrateObject.py
│   │   ├── capture
│   │   │   ├── detect.py
│   │   │   ├── hog_svm_detect.py
│   │   │   ├── url_video.py
│   │   │   └── video_coutours.py
│   │   ├── check_similar
│   │   │   ├── hamin_similar.py
│   │   │   ├── hash_similar.py
│   │   │   └── similary.py
│   │   ├── color
│   │   │   ├── colorcut.py
│   │   │   ├── colorDection.py
│   │   │   ├── colordetection_camera.py
│   │   │   ├── ColorExtractor.py
│   │   │   ├── colorMaskdetect.py
│   │   │   ├── colorMatching.py
│   │   │   ├── color_match.py
│   │   │   ├── ColorRecognition.py
│   │   │   └── selectColor.py
│   │   ├── contour
│   │   │   ├── cartoon.py
│   │   │   ├── prewitt.py
│   │   │   └── selectCanny.py
│   │   ├── enhancement.py
│   │   ├── features
│   │   │   ├── featureDetector
│   │   │   │   ├── agast.py
│   │   │   │   ├── akaze.py
│   │   │   │   ├── brisk.py
│   │   │   │   ├── detector_fast.py
│   │   │   │   ├── gftt.py
│   │   │   │   ├── harris_laplace.py
│   │   │   │   ├── harris_normal.py
│   │   │   │   ├── harris_subpixel.py
│   │   │   │   ├── kaze.py
│   │   │   │   ├── Lenna.png
│   │   │   │   ├── mser_color.py
│   │   │   │   ├── mser.py
│   │   │   │   ├── orb.py
│   │   │   │   ├── shi_tomasi.py
│   │   │   │   ├── simple_blob_detector.py
│   │   │   │   └── star.py
│   │   │   ├── featureDetect.py
│   │   │   ├── featureMatching
│   │   │   │   ├── BF
│   │   │   │   │   ├── boostdesc_matcher.py
│   │   │   │   │   ├── brief_matcher.py
│   │   │   │   │   ├── daisy_matcher.py
│   │   │   │   │   ├── freak_matcher.py
│   │   │   │   │   ├── latch_matcher.py
│   │   │   │   │   ├── lucid_matcher.py
│   │   │   │   │   └── vgg_matcher.py
│   │   │   │   ├── box_in_scene.png
│   │   │   │   ├── box.png
│   │   │   │   ├── Combined
│   │   │   │   │   ├── akaze_bf_matcher.py
│   │   │   │   │   ├── akaze_flann_matcher.py
│   │   │   │   │   ├── brisk_bf_matcher.py
│   │   │   │   │   ├── brisk_flann_matcher.py
│   │   │   │   │   ├── kaze_bf_matcher.py
│   │   │   │   │   ├── kaze_flann_matcher.py
│   │   │   │   │   └── ORB
│   │   │   │   │       ├── orb_bf_knn_matcher.py
│   │   │   │   │       ├── orb_bf_matcher.py
│   │   │   │   │       ├── orb_flann_knn_matcher.py
│   │   │   │   │       ├── orb_flann_matcher.py
│   │   │   │   │       └── orb_wta3_bf_matcher.py
│   │   │   │   └── FLANN
│   │   │   │       ├── boostdesc_knn_matcher.py
│   │   │   │       ├── boostdesc_matcher.py
│   │   │   │       ├── brief_knn_matcher.py
│   │   │   │       ├── brief_matcher.py
│   │   │   │       ├── daisy_knn_matcher.py
│   │   │   │       ├── daisy_matcher.py
│   │   │   │       ├── freak_knn_matcher.py
│   │   │   │       ├── freak_matcher.py
│   │   │   │       ├── latch_knn_matcher.py
│   │   │   │       ├── latch_matcher.py
│   │   │   │       ├── lucid_knn_matcher.py
│   │   │   │       ├── lucid_matcher.py
│   │   │   │       ├── vgg_knn_matcher.py
│   │   │   │       └── vgg_matcher.py
│   │   │   ├── featureMatching.py
│   │   │   └── match.py
│   │   ├── filr_image
│   │   │   ├── raw2rgb.py
│   │   │   ├── Temp_per_pixel.py
│   │   │   └── Transformer.jpg
│   │   ├── fix
│   │   │   └── inpaint_images.py
│   │   ├── image_utils
│   │   │   ├── box_cluster.py
│   │   │   ├── box.py
│   │   │   ├── convert.py
│   │   │   ├── cv2_utils.py
│   │   │   ├── data_augment.py
│   │   │   ├── pdf.py
│   │   │   ├── point.py
│   │   │   ├── rm_watermark.py
│   │   │   ├── SimHei.ttf
│   │   │   ├── similary.py
│   │   │   ├── table.py
│   │   │   ├── text_angle.py
│   │   │   ├── text.py
│   │   │   ├── utils.py
│   │   │   └── video.py
│   │   ├── img2gif.py
│   │   ├── imgtransform
│   │   │   ├── output.jpg
│   │   │   ├── RotationCorrection.py
│   │   │   ├── rotation.py
│   │   │   ├── scaleDown.py
│   │   │   ├── select_perspectiveTransform.py
│   │   │   └── used_perspectiveTransform.py
│   │   ├── kmeans
│   │   │   └── color_kmean.py
│   │   ├── light
│   │   │   └── light_adapt.py
│   │   ├── line
│   │   │   ├── extend_line.py
│   │   │   ├── fast_line_detection.py
│   │   │   ├── fitline.py
│   │   │   ├── horizental_parallel_line_detector.py
│   │   │   ├── line_cluster.py
│   │   │   ├── LineDetection.py
│   │   │   ├── Line_fit.py
│   │   │   └── line.py
│   │   ├── motionFlow
│   │   │   ├── dense_optical_flow.py
│   │   │   └── sparse_optical_flow.py
│   │   ├── pointcloud
│   │   │   ├── depth2pc.py
│   │   │   ├── geometry3d.py
│   │   │   ├── rgb2pcd.py
│   │   │   ├── timg
│   │   │   │   ├── color.jpg
│   │   │   │   ├── depth.png
│   │   │   │   ├── room_color.png
│   │   │   │   └── room_depth.png
│   │   │   ├── tpcd
│   │   │   │   ├── cube.pcd
│   │   │   │   ├── pc.ply
│   │   │   │   ├── room.pcd
│   │   │   │   └── scene.pcd
│   │   │   └── visualize_pcd.py
│   │   ├── ROI
│   │   │   ├── DP_ROI.py
│   │   │   ├── dst.jpg
│   │   │   └── selectROI.py
│   │   ├── segmentation
│   │   │   ├── approx_contours_and_convex_hull.png
│   │   │   ├── approx_contours_and_convex_hull.py
│   │   │   ├── blob_detection.jpg
│   │   │   ├── blob_detection.py
│   │   │   ├── blob_discrimination.png
│   │   │   ├── blob_discrimination.py
│   │   │   ├── convex_hull.png
│   │   │   ├── convex_hull.py
│   │   │   ├── Filtering-Blobs-using-OpenCV.png
│   │   │   ├── hierarchy_and_retrieval_mode.png
│   │   │   ├── hierarchy_and_retrieval_mode.py
│   │   │   ├── identify_shape.png
│   │   │   ├── identify_shape.py
│   │   │   ├── line_detection_hough.py
│   │   │   ├── line_detection.png
│   │   │   ├── line_detection_prob_hough.py
│   │   │   ├── matching_contour.png
│   │   │   ├── matching_contour.py
│   │   │   ├── matching_contour_template.png
│   │   │   ├── segmentation_and_contours.png
│   │   │   └── segmentation_and_contours.py
│   │   ├── skeleton
│   │   │   ├── morphologicalSkeleton.py
│   │   │   └── skeletonize.py
│   │   ├── subtractor
│   │   │   ├── BackgroundSubtractor.py
│   │   │   ├── camShift.py
│   │   │   ├── cut.py
│   │   │   ├── grabcut.py
│   │   │   ├── panelAbstractCut.py
│   │   │   └── SeparateFrontBack.py
│   │   ├── tech
│   │   │   ├── backgroung_remove_by_contour_corners_superb.py
│   │   │   ├── cropping_by_template_matching.py
│   │   │   ├── grabcut_using_mask.py
│   │   │   ├── max_filtering_and_colour_detection.py
│   │   │   ├── mean_square_error_two_images.py
│   │   │   ├── multiscale_template_matching.py
│   │   │   ├── segmentation_by_SLIC_and_extraction.py
│   │   │   ├── simple_labeling_ssim_hash_mse_chi.py
│   │   │   ├── subimage_template_matching.py
│   │   │   ├── template_alignment_ORB_based.py
│   │   │   ├── text_detction_by_contour_MSER.py
│   │   │   ├── text_detection_by_MSER_SWT.py
│   │   │   ├── text_detection_EAST_deep_learning.py
│   │   │   └── trapizoid_to_rectangle.py
│   │   ├── test.py
│   │   ├── threshold
│   │   │   ├── thresholdSegmentations.py
│   │   │   └── thresholdSplit.py
│   │   ├── tools
│   │   │   ├── enhancement.py
│   │   │   ├── image
│   │   │   │   └── test.png
│   │   │   ├── image_color.py
│   │   │   ├── image_enhancement.py
│   │   │   ├── image_filtering.py
│   │   │   ├── image_outline.py
│   │   │   ├── image_transformation.py
│   │   │   ├── main.py
│   │   │   └── utils.py
│   │   └── video
│   │       ├── getvideo.py
│   │       ├── image2video.py
│   │       ├── keyFrame.py
│   │       ├── stablizer.py
│   │       ├── videoprocess.py
│   │       └── videotools.py
│   ├── cvzone
│   │   ├── Examples
│   │   │   ├── CornerRectangleExample.py
│   │   │   ├── FaceDetectionExample.py
│   │   │   ├── FaceMeshExample.py
│   │   │   ├── FpsExample.py
│   │   │   ├── HandTrackingExample.py
│   │   │   ├── PoseEstimationExample.py
│   │   │   ├── SerialExample.py
│   │   │   └── StackImageExample.py
│   ├── label
│   │   ├── labeling.py
│   │   ├── labelmeUtils.py
│   │   ├── label.py
│   │   ├── labelvideo.py
│   │   ├── show_label_img.py
│   │   └── show_seglabel_img.py
│   ├── log
│   │   └── err_log
│   │       └── except_err.py
│   ├── math
│   │   ├── get_angle.py
│   │   ├── linemath.py
│   │   └── mathdistance.py
│   ├── network
│   │   ├── baiduSpider.py
│   │   ├── Crawler_downloads_Baidu_pictures.py
│   │   ├── datawash.py
│   │   ├── download_image_one.py
│   │   └── requesetSpider.py
├── docs
├── images
│   └── lena.png
├── img
├── LICENSE
├── package
├── README.md
├── requirements.txt
├── temptest
│   ├── addmask.py
│   ├── anpd_input.jpg
│   ├── anpd.py
│   ├── contourdetection.py
│   ├── convexHull.py
│   ├── crop.py
│   ├── detectpose.py
│   ├── graph_opt.pb
│   ├── lifegames.py
│   ├── line_skeletonize.py
│   ├── remove_water_mask.py
│   ├── remove_water_print.py
│   ├── result.py
│   └── rm_watermark.py
├── testfunc.py
├── tools
│   ├── cut_label_image.py
│   └── XML_tran_coco.py
└── utils
    ├── FileHelper.py
    ├── ImgFileHelper.py
    ├── ImgHelper.py
    ├── __init__.py
    ├── JsonHelper.py
    ├── NpyHelper.py
    ├── PklHelper.py
    └── VideoreadHelper.py

```





## 五、引用
