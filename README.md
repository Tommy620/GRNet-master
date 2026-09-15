# GRNet

## Installation
---
We run the project on CUDA 11.3 (PyTorch 1.11.0, Python 3.7).
#### **Step 1.** Create a conda virtual environment and activate it
```
conda create -n grnet python=3.7 -y
conda activate grnet
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

#### **Step 2.** Install GRNet
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


## Training
Change the `data_folder` and `checkpoints` in `train_with_sacred.py` before running:
```
python train_with_sacred.py
```
Iterative training uses `max_r` / `max_t` (e.g. 5.0°/0.5m, 2.0°/0.2m, 1.0°/0.1m).

## Testing
Change the `data_folder`, `output` and `weights` in `evaluate_calib.py` before running:
```
python evaluate_calib.py
```

# Contact
For questions about our paper or code, please contact wangzi@gml.ac.cn.
