# GRNet

## Installation
---
we run the project on CUDA11.8
#### **Step 1.** Create a conda virtual environment and activate it
```
conda create -n grnet python=3.7 -y
conda activate grnet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.8 -c pytorch
```

#### **Step 2.** Install IFNet
```
git clone https://github.com/Tommy620/GRNet-master.git
cd GRNet-master
pip install -r requirements.txt
```


#### **Step 3.** Download the weights from https://pan.baidu.com/s/1T3q1Ls6NtdF9D5UPBtVzng?pwd=IFNT Extract code: IFNT 


#### Data Preparation
Dataset:KITTI dataset; Color.
The data folders are organized as follows:
```
├── data/
|   └── sequences
|       └── 00  
|           └── image_2
|               └── 000000.png
|               └── 000001.png
|               └── ...
|           └──velodyne
|               └── 000000.bin
|               └── 000001.bin
|               └── ...
```


## Testing
Change the 'data_folder', 'output' and 'weights' in `evaluate_calib.py` before running:
```
python evaluate_calib.py
```

# Contact
For questions about our paper or code, please contact wangzi@gml.ac.cn.
