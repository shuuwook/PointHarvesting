# **PointHarvesting**

>This repository **PointHarvesting** is for _**Wasserstein Distributional Harvesting for Highly Dense 3D Point Clouds**_ paper
___

## [ Paper ]
[_Wasserstein Distributional Harvesting for Highly Dense 3D Point Clouds_](~~~~)  
(Dong Wook Shu, Sung Woo Park, Junseok Kwon)
___


## [Results]
- We used [MeshLab](http://www.meshlab.net/) to render outputs.

- To open outputs in MeshLab, you can use './utils/save_ply.py'.  
It transform numpy array into ply file.

- Plane.  
![Plane progressive Sampling](https://github.com/seowok/PointHarvesting/blob/master/results/plane_progressive.gif)

- Chair.  
![Chair progressive Sampling](https://github.com/seowok/PointHarvesting/blob/master/results/chair_progressive.gif)  

- Car.  
![Car progressive Sampling](https://github.com/seowok/PointHarvesting/blob/master/results/car_progressive.gif) 
___

## [Evaluation]
- We used evaluation metrics for 3D point cloud generation which is proposed in [PointFlow](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_PointFlow_3D_Point_Cloud_Generation_With_Continuous_Normalizing_Flows_ICCV_2019_paper.pdf).

- We used train split in ShapeNet.Core.v2 dataset for training our model.

- We used valid split in ShapeNet.Core.v2 dataset for evaluation. (It is according to the setting of the PointFlow.)
___
                           
           
## [Setting]
This project was tested on **Ubuntu 16.04** / **RTX2080ti** / **CUDA-10.0**
Using _conda install_ command is recommended to setting.

### Packages
- Python 3.6
- Numpy
- Pytorch with cudatoolkit=10.0 (Recommened : conda install pytorch torchvision cudatoolkit=10.0 -c pytorch -y)
- visdom
- tqdm

### Evaluation
- Evaluation packages are tested only in **Ubuntu** setting.
- We included bash file for installing packages from [PointFlow](https://github.com/stevenygd/PointFlow)
- The compile of CUDA version for CD,EMD is only avaliable for cuda 10.0
- CUDA 10.0 is not supported to RTX30 series.
___

## [Arguments]
In our project, **arguments.py** file has almost every parameters to specify for model.
