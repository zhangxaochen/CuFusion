
# CuFusion #
This repo is a fork of [StanfordPCL(Qianyi Zhou's PCL Fork)](https://github.com/qianyizh/StanfordPCL "qianyizh/StanfordPCL") containing the code implementation of our work [**CuFusion**](https://www.preprints.org/manuscript/201708.0022/v1), a novel approach for  accurate real-time depth camera tracking and volumetric scene reconstruction with a known cuboid reference object.

This is the initial version of our algorithm with trial dirty code and redundant comments.

# Dataset #
We introduce a dataset called [**CU3D**](https://drive.google.com/open?id=0B4vahSr3aGadN0ozUmE3dVNSXzA), for the validation of our algorithm. 

The dataset contains 3 noiseless synthetic sequences with both the ground-truth (GT) camera trajectories and GT mesh scene models, and 6 noisy real-world scanning sequences with ONLY the GT mesh models of the scanned objectives.

**Note:** Different from the [**TUM RGB-D dataset**](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps), where the depth images are scaled by a factor of 5000, currently our depth values are stored in the PNG files in millimeters, namely, with a scale factor of 1000. We may remake the data to conform to the style of the TUM dataset later.

# Related Publications #

If you find the **code** or [**CU3D dataset**](https://drive.google.com/open?id=0B4vahSr3aGadN0ozUmE3dVNSXzA) valuable for your research please cite this work:
> ZHANG, C.; Hu, Y. [CuFusion: Accurate Real©\time Camera Tracking and Volumetric Scene Reconstruction with a Cuboid](https://www.preprints.org/manuscript/201708.0022/v1). Preprints 2017, 2017080022 (doi: 10.20944/preprints201708.0022.v1).

# How to build #
We've tested our code on Windows 10, with Visual Studio 2010 (Though other configurations may work)
To build this repo from source, you should follow the [instructions from PCL](https://github.com/PointCloudLibrary/pcl#compiling), e.g., [Compiling PCL from source on Windows](http://www.pointclouds.org/documentation/tutorials/compiling_pcl_windows.php).
If you've already built/installed PCL, you need to build the [pcl_gpu_kinfu_large_scale] library and [pcl_kinfu_largeScale] executable to run our algorithm.

Additional prerequisites:
- [PEAC]: our fork of [the work of Chen Feng](http://simbaforrest.github.io/blog/2015/10/16/peac.html). Clone our repo and add the source code manually to [pcl_kinfu_largeScale] and [pcl_gpu_kinfu_large_scale] to avoid compilation errors.
- OpenCV: for image processing; we tested our code with version 2.4.x;
- Cuda

[peac]: https://github.com/zhangxaochen/peac
[pcl_kinfu_largeScale]: https://github.com/zhangxaochen/CuFusion/tree/master/gpu/kinfu_large_scale/tools
[pcl_gpu_kinfu_large_scale]: https://github.com/zhangxaochen/CuFusion/tree/master/gpu/kinfu_large_scale

# How to use #
##1. Our algorithm
To test our algorithm, run `pcl_kinfu_largeScale_release.exe` in command line:

```
pcl_kinfu_largeScale_release.exe --shift_z -0.3 -cusz 400,300,250 -e2cDist 0.05 -trunc_dist 0.025 -vs 1 -tsdfVer 11 --camera myKinectV1.param -eval "04-lambunny\"
```
Params explanation:
- `--shift_z <in_meters>`: initial shift along z axis
- `-cusz <a,b,c>`: toggle to use our algorithm, with the reference cuboid size being `a*b*c` millimeters
- `-e2cDist <in_meters>`: edge-to-edge distance threshold for correspondence matching
- `-trunc_dist <in_meters>`: truncation distance of the TSDF
- `-vs <in_meters>`: volume size of the reconstruction
- `-tsdfVer <version number>`: different versions of our TSDF fusion strategy, currently stable with version `11`
- `--camera <param_file>`: launch parameters from the file, containing the camera intrinsic matrix
- `-eval <eval_folder>`: folder containing the evaluation dataset.

## 2. Algorithms for comparison in our work
### 2.1. KinectFusion [1]

```
pcl_kinfu_largeScale_release.exe  --shift_z -0.3 -trunc_dist 0.025 -vs 1 --camera myKinectV1.param -eval E:\geo-cu399-pen\turntable\antenna2c-raw_frames-ts
```

### 2.2. Zhou et al. [2]

```
pcl_kinfu_largeScale_release.exe --shift_z -0.3 --bdr_odometry -trunc_dist 0.025 -vs 1 --camera myKinectV1.param -eval "04-lambunny\"
```

### 2.3. ElasticFusion [3]
To test this algorithm, we need to convert the image sequences to  `*.klg` files. We use [**this code**](https://github.com/HTLife/png_to_klg) for such work, build [**ElasticFusion**](https://github.com/mp3guy/ElasticFusion) from source, and run it.


# References
1. Newcombe, R. A.; Izadi, S.; Hilliges, O.; Molyneaux, D.; Kim, D.; Davison, A. J.; Kohli, P.; Shotton, J.; 				Hodges, S.; Fitzgibbon, A. KinectFusion: Real-time Dense Surface Mapping and Tracking. In Proceedings of the 2011 10th IEEE International Symposium on Mixed and Augmented Reality; ISMAR ¡¯11; IEEE Computer Society: Washington, DC, USA, 2011; pp. 127¨C136.
2. Zhou, Q.-Y.; Koltun, V. Depth camera tracking with contour cues. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR); 2015; pp. 632¨C638.
3. Whelan, T.; Leutenegger, S.; Moreno, R. S.; Glocker, B.; Davison, A. ElasticFusion: Dense SLAM Without A Pose Graph. In Proceedings of Robotics: Science and Systems; Rome, Italy, 2015.



# Original README.txt of StanfordPCL
> ===============================================================================
> =                          Qianyi Zhou's PCL Fork                             =
> +==============================================================================
> 
> I have been receiving requests for the source code of 
> pcl_kinfu_largeScale_release.exe, which is a critical module in the robust
> scene reconstruction system we have developed.
> 
> Project: http://qianyi.info/scene.html
> Code: https://github.com/qianyizh/ElasticReconstruction
> Executable system: http://redwood-data.org/indoor/tutorial.html
> 
> Thus I publish my fork of PCL here as a reference.
> 
> ===============================================================================
> 
> Disclaimer
> 
> I forked PCL from an early development version three years ago, and have made
> numerous changes. This repository is an image of my personal development chunk.
> It contains many experimental functions and redundant code.
> 
> THERE IS NO GUARANTEE THIS CODE CAN BE COMPILED OR USED. WE DO NOT INTEND TO
> PROVIDE ANY SUPPORT FOR THIS CODE. IT SHOULD BE USED ONLY AS A REFERENCE.
> 
> If you are looking for the official PCL, go to this repository:
> https://github.com/PointCloudLibrary/pcl
> 
> ===============================================================================
> 
> License
> 
> As part of the scene reconstruction system, the code of this repository is
> released under the MIT license.
> 
> Some parts of this repository are from different libraries:
> 
> Original PCL - BSD license
> SuiteSparse - LGPL3+ license (we have a static library linked by Visual Studio)
> Eigen - MPL2 license (we have a copy of a certain version of Eigen)



