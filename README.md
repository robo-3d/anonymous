# Robo3D: Towards Robust and Reliable 3D Perception against Corruptions
`Robo3D` is an evaluation suite heading toward robust and reliable 3D perception in autonomous driving. With it, we probe the **robustness** of 3D **detectors** and **segmentors** under out-of-distribution (OoD) scenarios against **corruptions** that occur in the real-world environment. Specifically, we consider natural corruptions happen in the following cases:
1. **Adverse weather conditions**, such as `fog`, `wet ground`, and `snow`;
2. **External disturbances** that are caused by `motion blur` or result in LiDAR `beam missing`;
3. **Internal sensor failure**, including `crosstalk`, possible `incomplete echo`, and `cross-sensor` scenarios.

| | | |
| :---: | :---: | :---: |
| <img src="docs/figs/teaser/clean.png" width="240"> | <img src="docs/figs/teaser/fog.png" width="240"> | <img src="docs/figs/teaser/wet_ground.png" width="240"> |
| **Clean** | **Fog** | **Wet Ground** |
| <img src="docs/figs/teaser/snow.png" width="240"> | <img src="docs/figs/teaser/motion_blur.png" width="240"> | <img src="docs/figs/teaser/beam_missing.png" width="240">
| **Snow** | **Motion Blur** | **Beam Missing** |
| <img src="docs/figs/teaser/crosstalk.png" width="240"> | <img src="docs/figs/teaser/incomplete_echo.png" width="240"> | <img src="docs/figs/teaser/cross_sensor.png" width="240"> | 
| **Crosstalk** | **Incomplete Echo** | **Cross-Sensor** |
| | | |


## Outline
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
- [Benchmark](#benchmark)
- [Evaluation on Robo3D: 3D Segmentors](#evaluation-on-robo3d-3d-segmentors)
- [Evaluation on Robo3D: 3D Detectors](#evaluation-on-robo3d-3d-detectors)
- [Create Corruption Set](#create-corruption-set)
- [License](#license)



## Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for the installation details.

## Data Preparation
```
└── data 
    └── sets
         │── Robo3D      
         │   │── KITTI-C                
         │   │    │── beam_missing       
         │   │    │── crosstalk
         │   │    │── ...  
         │   │    └── wet_ground
         │   │── SemanticKITTI-C    
         │   │── nuScenes-C
         │   └── WOD-C                                                                                   
         │── KITTI          
         │── SemanticKITTI       
         │── nuScenes                
         └── WOD            
```

Please refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for the details to prepare the datasets.


## Model Zoo

<details open>
<summary>&nbsp<b>LiDAR Semantic Segmentation</b></summary>

> - [x] **[SqueezeSeg](https://arxiv.org/abs/1710.07368), ICRA 2018.** <sup>[**`[Code]`**](https://github.com/BichenWuUCB/SqueezeSeg)</sup>
> - [x] **[SqueezeSegV2](https://arxiv.org/abs/1809.08495), ICRA 2019.** <sup>[**`[Code]`**](https://github.com/xuanyuzhou98/SqueezeSegV2)</sup>
> - [x] **[MinkowskiNet](https://arxiv.org/abs/1904.08755), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/NVIDIA/MinkowskiEngine)</sup>
> - [x] **[RangeNet++](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf), IROS 2019.** <sup>[**`[Code]`**](https://github.com/PRBonn/lidar-bonnetal)</sup>
> - [x] **[KPConv](https://arxiv.org/abs/1904.08889), ICCV 2019.** <sup>[**`[Code]`**](https://github.com/HuguesTHOMAS/KPConv)</sup>
> - [x] **[SalsaNext](https://arxiv.org/abs/2003.03653), ISVC 2020.** <sup>[**`[Code]`**](https://github.com/TiagoCortinhal/SalsaNext)</sup>
> - [x] **[PolarNet](https://arxiv.org/abs/2003.14032), CVPR 2020.** <sup>[**`[Code]`**](https://github.com/edwardzhou130/PolarSeg)</sup>
> - [x] **[SPVCNN](https://arxiv.org/abs/2007.16100), ECCV 2020.** <sup>[**`[Code]`**](https://github.com/mit-han-lab/spvnas)</sup>
> - [x] **[Cylinder3D](https://arxiv.org/abs/2011.10033), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/xinge008/Cylinder3D)</sup>
> - [x] **[FIDNet](https://arxiv.org/abs/2109.03787), IROS 2021.** <sup>[**`[Code]`**](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI)</sup>
> - [x] **[RPVNet](https://arxiv.org/abs/2103.12978), ICCV 2021.**
> - [x] **[CENet](https://arxiv.org/abs/2207.12691), ICME 2022.** <sup>[**`[Code]`**](https://github.com/huixiancheng/CENet)</sup>
> - [x] **[CPGNet](https://arxiv.org/abs/2204.09914), ICRA 2022.** <sup>[**`[Code]`**](https://github.com/GangZhang842/CPGNet)</sup>
> - [x] **[2DPASS](https://arxiv.org/abs/2207.04397), ECCV 2022.** <sup>[**`[Code]`**](https://github.com/yanx27/2DPASS)</sup>
> - [x] **[GFNet](https://arxiv.org/abs/2207.02605), TMLR 2022.** <sup>[**`[Code]`**](https://github.com/haibo-qiu/GFNet)</sup>
> - [x] **[PIDS](https://arxiv.org/abs/2211.15759), WACV 2023.** <sup>[**`[Code]`**](https://github.com/lordzth666/WACV23_PIDS-Joint-Point-Interaction-Dimension-Search-for-3D-Point-Cloud)</sup>
> - [x] **[WaffleIron](http://arxiv.org/abs/2301.10100), arXiv 2023.** <sup>[**`[Code]`**](https://github.com/valeoai/WaffleIron)</sup>

</details>

<details open>
<summary>&nbsp<b>3D Object Detection</b></summary>

> - [x] **[SECOND](https://www.mdpi.com/1424-8220/18/10/3337), Sensors 2018.** <sup>[**`[Code]`**](https://github.com/traveller59/second.pytorch)</sup>
> - [x] **[PointPillars](https://arxiv.org/abs/1812.05784), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/nutonomy/second.pytorch)</sup>
> - [x] **[PointRCNN](https://arxiv.org/abs/1812.04244), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/sshaoshuai/PointRCNN)</sup>
> - [x] **[Part-A2](https://arxiv.org/abs/1907.03670), T-PAMI 2020.**
> - [x] **[PV-RCNN](https://arxiv.org/abs/1912.13192), CVPR 2020.** <sup>[**`[Code]`**](https://github.com/sshaoshuai/PV-RCNN)</sup>
> - [x] **[CenterPoint](https://arxiv.org/abs/2006.11275), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/tianweiy/CenterPoint)</sup>
> - [x] **[PV-RCNN++](https://arxiv.org/abs/2102.00463), IJCV 2022.** <sup>[**`[Code]`**](https://github.com/open-mmlab/OpenPCDet)</sup>

</details>


## Benchmark

### LiDAR Semantic Segmentation

The *mean Intersection-over-Union (mIoU)* is consistently used as the main indicator for evaluating model performance in our  LiDAR semantic segmentation benchmark. The following two metrics are adopted to compare between models' robustness:
- **mCE (the lower the better):** The *average corruption error* (in percentage) of a candidate model compared to the baseline model, which is calculated among all corruption types across three severity levels.
- **mRR (the higher the better):** The *average resilience rate* (in percentage) of a candidate model compared to its "clean" performance, which is calculated among all corruption types across three severity levels.


### :red_car:&nbsp; SemanticKITTI-C

<p align="center">
  <img src="docs/figs/stat/metrics_semkittic.png" align="center" width="100%">
</p>

| Model | mCE (%) | mRR (%) | Clean | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [SqueezeSeg](docs/results/SqueezeSeg.md) | 164.87 | 66.81 | 31.61 | 18.85 | 27.30 | 22.70 | 17.93 | 25.01 | 21.65 | 27.66 | 7.85 |
| [SqueezeSegV2](docs/results/SqueezeSegV2.md) | 152.45 | 65.29 | 41.28 | 25.64 | 35.02 | 27.75 | 22.75 | 32.19 | 26.68 | 33.80 | 11.78 |
| [RangeNet<sub>21</sub>](docs/results/RangeNet-dark21.md) | 136.33 | 73.42 | 47.15 | 31.04 | 40.88 | 37.43 | 31.16 | 38.16 | 37.98 | 41.54 | 18.76 |
| [RangeNet<sub>53</sub>](docs/results/RangeNet-dark21.md) | 130.66 | 73.59 | 50.29 | 36.33 | 43.07 | 40.02 | 30.10 | 40.80 | 46.08 | 42.67 | 16.98 |
| [SalsaNext](docs/results/SalsaNext.md) | 116.14 | 80.51 | 55.80 | 34.89 | 48.44 | 45.55 | 47.93 | 49.63 | 40.21 | 48.03 | 44.72 |
| [FIDNet<sub>34</sub>](docs/results/FIDNet.md) | 113.81 | 76.99 | 58.80 | 43.66 | 51.63 | 49.68 | 40.38 | 49.32 | 49.46 | 48.17 | 29.85 |
| [CENet<sub>34</sub>](docs/results/CENet.md) | 103.41 | 81.29 | 62.55 | 42.70 | 57.34 | 53.64 | 52.71 | 55.78 | 45.37 | 53.40 | 45.84 |
| |
| [KPConv](docs/results/KPConv.md) | 99.54 | 82.90 | 62.17 | 54.46 | 57.70 | 54.15 | 25.70 | 57.35 | 53.38 | 55.64 | 53.91 |
| [PIDS<sub>NAS1.25x</sub>]() | 104.13 | 77.94 | 63.25 | 47.90 | 54.48 | 48.86 | 22.97 | 54.93 | 56.70 | 55.81 | 52.72 |
| [PIDS<sub>NAS2.0x</sub>]()  | 101.20 | 78.42 | 64.55 | 51.19 | 55.97 | 51.11 | 22.49 | 56.95 | 57.41 | 55.55 | 54.27 |
| [WaffleIron](docs/results/WaffleIron.md) | 109.54 | 72.18 | 66.04 | 45.52 | 58.55 | 49.30 | 33.02 | 59.28 | 22.48 | 58.55 | 54.62 |
| |
| [PolarNet](docs/results/PolarNet.md) | 118.56 | 74.98 | 58.17 | 38.74 | 50.73 | 49.42 | 41.77 | 54.10 | 25.79 | 48.96 | 39.44 |
| |
| <sup>:star:</sup>[MinkUNet<sub>18</sub>](docs/results/MinkUNet-18_cr1.0.md) | 100.00 | 81.90 | 62.76 | 55.87 | 53.99 | 53.28 | 32.92 | 56.32 | 58.34 | 54.43 | 46.05 |
| [MinkUNet<sub>34</sub>](docs/results/MinkUNet-34_cr1.6.md) | 100.61 | 80.22 | 63.78 | 53.54 | 54.27 | 50.17 | 33.80 | 57.35 | 58.38 | 54.88 | 46.95 |
| [Cylinder3D<sub>SPC</sub>](docs/results/Cylinder3D.md) | 103.25 | 80.08 | 63.42 | 37.10 | 57.45 | 46.94  | 52.45 | 57.64 | 55.98 | 52.51 | 46.22 |
| [Cylinder3D<sub>TSC</sub>](docs/results/Cylinder3D-TS.md) | 103.13 | 83.90 | 61.00 | 37.11 | 53.40 | 45.39 | 58.64 | 56.81 | 53.59 | 54.88 | 49.62 |
| |
| [SPVCNN<sub>18</sub>](docs/results/SPVCNN-18_cr1.0.md) | 100.30 | 82.15 | 62.47 | 55.32 | 53.98 | 51.42 | 34.53 | 56.67 | 58.10 | 54.60 | 45.95 |
| [SPVCNN<sub>34</sub>](docs/results/SPVCNN-34_cr1.6.md) | 99.16 | 82.01 | 63.22 | 56.53 | 53.68 | 52.35 | 34.39 | 56.76 | 59.00 | 54.97 | 47.07 |
| [RPVNet](docs/results/RPVNet.md) | 111.74 | 73.86 | 63.75 | 47.64 | 53.54 | 51.13 | 47.29 | 53.51 | 22.64 | 54.79 | 46.17 |
| [CPGNet]() | 107.34 | 81.05 | 61.50 | 37.79 | 57.39 | 51.26 | 59.05 | 60.29 | 18.50 | 56.72 | 57.79 |
| [2DPASS](docs/results/DPASS.md) | 106.14 | 77.50 | 64.61 | 40.46 | 60.68 | 48.53 | 57.80 | 58.78 | 28.46 | 55.84 | 50.01 |
| [GFNet](docs/results/GFNet.md) | 108.68 | 77.92 | 63.00 | 42.04 | 56.57 | 56.71 | 58.59 | 56.95 | 17.14 | 55.23 | 49.48 |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### :blue_car:&nbsp; nuScenes-C

<p align="center">
  <img src="docs/figs/stat/metrics_nusc_seg.png" align="center" width="100%">
</p>

| Model | mCE (%) | mRR (%) | Clean | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [FIDNet<sub>34</sub>](docs/results/FIDNet.md) | 122.42 | 73.33 | 71.38 | 64.80 | 68.02 | 58.97 | 48.90 | 48.14 | 57.45 | 48.76 | 23.70 | 
| [CENet<sub>34</sub>](docs/results/CENet.md) | 112.79 | 76.04 | 73.28 | 67.01 | 69.87 | 61.64 | 58.31 | 49.97 | 60.89 | 53.31 | 24.78 |
| |
| [WaffleIron](docs/results/WaffleIron.md) | 106.73 | 72.78 | 76.07 | 56.07 | 73.93 | 49.59 | 59.46 | 65.19 | 33.12 | 61.51 | 44.01 |
| |
| [PolarNet](docs/results/PolarNet.md) | 115.09 | 76.34 | 71.37 | 58.23 | 69.91 | 64.82 | 44.60 | 61.91 | 40.77 | 53.64 | 42.01 |
| |
| <sup>:star:</sup>[MinkUNet<sub>18</sub>](docs/results/MinkUNet-18_cr1.0.md) | 100.00 | 74.44 | 75.76 | 53.64 | 73.91 | 40.35 | 73.39 | 68.54 | 26.58 | 63.83 | 50.95 |
| [MinkUNet<sub>34</sub>](docs/results/MinkUNet-34_cr1.6.md) | 96.37 | 75.08 | 76.90 | 56.91 | 74.93 | 37.50 | 75.24 | 70.10 | 29.32 | 64.96 | 52.96 |
| [Cylinder3D<sub>SPC</sub>](docs/results/Cylinder3D.md) | 111.84 | 72.94 | 76.15 | 59.85 | 72.69 | 58.07 | 42.13 | 64.45 | 44.44 | 60.50 | 42.23 |
| [Cylinder3D<sub>TSC</sub>](docs/results/Cylinder3D-TS.md) | 105.56 | 78.08 | 73.54 | 61.42 | 71.02 | 58.40 | 56.02 | 64.15 | 45.36 | 59.97 | 43.03 |
| |
| [SPVCNN<sub>18</sub>](docs/results/SPVCNN-18_cr1.0.md) | 106.65 | 74.70 | 74.40 | 59.01 | 72.46 | 41.08 | 58.36 | 65.36 | 36.83 | 62.29 | 49.21 |
| [SPVCNN<sub>34</sub>](docs/results/SPVCNN-34_cr1.6.md) | 97.45 | 75.10 | 76.57 | 55.86 | 74.04 | 41.95 | 74.63 | 68.94 | 28.11 | 64.96 | 51.57 |
| [2DPASS](docs/results/DPASS.md) | 98.56 | 75.24 | 77.92 | 64.50 | 76.76 | 54.46 | 62.04 | 67.84 | 34.37 | 63.19 | 45.83 |
| [GFNet](docs/results/GFNet.md) | 92.55 | 83.31 | 76.79 | 69.59 | 75.52 | 71.83 | 59.43 | 64.47 | 66.78 | 61.86 | 42.30 |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### :taxi:&nbsp; WOD-C

<p align="center">
  <img src="docs/figs/stat/metrics_wod_seg.png" align="center" width="100%">
</p>

| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| <sup>:star:</sup>[MinkUNet<sub>18</sub>](docs/results/MinkUNet-18_cr1.0.md) | 100.00 | 91.22 | 69.06 | 66.99 | 60.99 | 57.75 | 68.92 | 64.15 | 65.37 | 63.36 | 56.44 |
| [MinkUNet<sub>34</sub>](docs/results/MinkUNet-34_cr1.6.md) | 96.21 | 91.80 | 70.15 | 68.31 | 62.98 | 57.95 | 70.10 | 65.79 | 66.48 | 64.55 | 59.02 | 
| [Cylinder3D<sub>TSC</sub>](docs/results/Cylinder3D-TS.md) | 106.02 | 92.39 | 65.93 | 63.09 | 59.40 | 58.43 | 65.72 | 62.08 | 62.99 | 60.34 | 55.27 |
| |
| [SPVCNN<sub>18</sub>](docs/results/SPVCNN-18_cr1.0.md) | 103.60 | 91.60 | 67.35 | 65.13 | 59.12 | 58.10 | 67.24 | 62.41 | 65.46 | 61.79 | 54.30 |
| [SPVCNN<sub>34</sub>](docs/results/SPVCNN-34_cr1.6.md) | 98.72 | 92.04 | 69.01 | 67.10 | 62.41 | 57.57 | 68.92 | 64.67 | 64.70 | 64.14 | 58.63 |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### 3D Object Detection

The *mean average precision (mAP)* and *nuScenes detection score (NDS)* are consistently used as the main indicator for evaluating model performance in our  LiDAR semantic segmentation benchmark. The following two metrics are adopted to compare between models' robustness:
- **mCE (the lower the better):** The *average corruption error* (in percentage) of a candidate model compared to the baseline model, which is calculated among all corruption types across three severity levels.
- **mRR (the higher the better):** The *average resilience rate* (in percentage) of a candidate model compared to its "clean" performance, which is calculated among all corruption types across three severity levels.


### :red_car:&nbsp; KITTI-C

<p align="center">
  <img src="docs/figs/stat/metrics_kittic.png" align="center" width="100%">
</p>

| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [PointPillars]() | 110.67 | 74.94 | 66.70 | 45.70 | 66.71 | 35.77 | 47.09 | 52.24 | 60.01 | 54.84 | 37.50 |
| [SECOND]() | 95.93 | 82.94 | 68.49 | 53.24 | 68.51 | 54.92 | 49.19 | 54.14 | 67.19 | 59.25 | 48.00 |
| [PointRCNN]() | 91.88 | 83.46 | 70.26 | 56.31 | 71.82 | 50.20 | 51.52 | 56.84 | 65.70 | 62.02 | 54.73 |
| [PartA2<sub>Free</sub>]() | 82.22 | 81.87 | 76.28 | 58.06 | 76.29 | 58.17 | 55.15 | 59.46 | 75.59 | 65.66 | 51.22 |
| [PartA2<sub>Anchor</sub>]() | 88.62 | 80.67 | 73.98 | 56.59 | 73.97 | 51.32 | 55.04 | 56.38 | 71.72 | 63.29 | 49.15 |
| [PVRCNN]() | 90.04 | 81.73 | 72.36 | 55.36 | 72.89 | 52.12 | 54.44 | 56.88 | 70.39 | 63.00 | 48.01 |
| <sup>:star:</sup>[CenterPoint]() | 100.00 | 79.73 | 68.70 | 53.10 | 68.71 | 48.56 | 47.94 | 49.88 | 66.00 | 58.90 | 45.12 |
| [SphereFormer]() | - | - | - | - | - | - | - | - | - | - | - |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### :blue_car:&nbsp; nuScenes-C

<p align="center">
  <img src="docs/figs/stat/metrics_nusc_det.png" align="center" width="100%">
</p>

| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [PointPillars<sub>MH</sub>]() | 102.90 | 77.24 | 43.33 | 33.16 | 42.92 | 29.49 | 38.04 | 33.61 | 34.61 | 30.90 | 25.00 |
| [SECOND<sub>MH</sub>]() | 97.50 | 76.96 | 47.87 | 38.00 | 47.59 | 33.92 | 41.32 | 35.64 | 40.30 | 34.12 | 23.82 |
| <sup>:star:</sup>[CenterPoint]() | 100.00 | 76.68 | 45.99 | 35.01 | 45.41 | 31.23 | 41.79 | 35.16 | 35.22 | 32.53 | 25.78 |
| [CenterPoint<sub>LR</sub>]() | 98.74 | 72.49 | 49.72 | 36.39 | 47.34 | 32.81 | 40.54 | 34.47 | 38.11 | 35.50 | 23.16 |
| [CenterPoint<sub>HR</sub>]() | 95.80 | 75.26 | 50.31 | 39.55 | 49.77 | 34.73 | 43.21 | 36.21 | 40.98 | 35.09 | 23.38 |
| [SphereFormer]() | - | - | - | - | - | - | - | - | - | - | - |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### :taxi:&nbsp; WOD-C

<p align="center">
  <img src="docs/figs/stat/metrics_wod_det.png" align="center" width="100%">
</p>

| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [PointPillars]() | 127.53 | 81.23 | 50.17 | 31.24 | 49.75 | 46.07 | 34.93 | 43.93 | 39.80 | 43.41 | 36.67  | 
| [SECOND]() | 121.43 | 81.12 | 53.37 | 32.89 | 52.99 | 47.20 | 35.98 | 44.72 | 49.28 | 46.84 | 36.43 | 
| [PVRCNN]() | 104.90 | 82.43 | 61.27 | 37.32 | 61.27 | 60.38 | 42.78 | 49.53 | 59.59 | 54.43 | 38.73 |
| <sup>:star:</sup>[CenterPoint]() | 100.00 | 83.30 | 63.59 | 43.06 | 62.84 | 58.59 | 43.53 | 54.41 | 60.32 | 57.01 | 43.98 |
| [PVRCNN++]() | 91.60 | 84.14 | 67.45 | 45.50 | 67.18 | 62.71 | 47.35 | 57.83 | 64.71 | 60.96 | 47.77 |
| [SphereFormer]() | - | - | - | - | - | - | - | - | - | - | - |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.

  

## Evaluation on Robo3D: 3D Segmentors
We provide runable scripts for evaluating 3D segmentors as follows.

#### SemanticKITTI-C
```
sh slurm_infer.sh ${PARTITION}  ${JOB_NAME} 1 --cfg_file ./tools/cfgs/voxel/semantic_kitti/minkunet_mk18_cr10.yaml --pretrained_model xxx/xxx.pth
```
Or
```
CUDA_VISIBLE_DEVICES=0 sh dist_infer.sh  1 --cfg_file ./tools/cfgs/voxel/semantic_kitti/minkunet_mk18_cr10.yaml --pretrained_model xxx/xxx.pth
```

#### nuScenes-C
```
sh slurm_infer.sh ${PARTITION}  ${JOB_NAME} 1 --cfg_file ./tools/cfgs/voxel/nuscenes/minkunet_mk18_cr10.yaml --pretrained_model xxx/xxx.pth
```
Or
```
CUDA_VISIBLE_DEVICES=0 sh dist_infer.sh 1 --cfg_file ./tools/cfgs/voxel/nuscenes/minkunet_mk18_cr10.yaml --pretrained_model xxx/xxx.pth
```

#### WOD-C
```
sh slurm_infer.sh ${PARTITION}  ${JOB_NAME} 1  --cfg_file ./tools/cfgs/voxel/waymo/minkunet_mk18_cr10.yaml --pretrained_model xxx/xxx.pth
```
Or
```
CUDA_VISIBLE_DEVICES=0 sh dist_infer.sh 1 --cfg_file ./tools/cfgs/voxel/waymo/minkunet_mk18_cr10.yaml --pretrained_model xxx/xxx.pth
```

## Evaluation on Robo3D: 3D Detectors

We provide evaluation examples for 3D detectors based on the OpenPCDet pipeline.

Simply modify `OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py` allows us to run the evaluation on detection datasets. The modification examples are as follows.

```python
class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, corruptions, severity, training=True, root_path=None, logger=None ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        
    # load corruption type from script, self.corruptions[0] save lidar corruption type
    MAP = {
    'fog': fog,
    ...
    }
    
    # for lidar 
    def get_lidar(self, idx):
        try:
            if "bbox" in self.corruptions[0]:
                data = MAP[self.corruptions[0]](data,self.severity,idx)
            else:
                data = MAP[self.corruptions[0]](data,self.severity)
        except:
            pass
        
        return data
```

## Create Corruption Set
You can manage to create your own "Robo3D" corrpution sets on other LiDAR-based point cloud datasets using our defined corruption types! Follow the instructions listed in [CREATE.md](docs/CREATE.md).

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>, while some specific operations in this codebase might be with other licenses.

