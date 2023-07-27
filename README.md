# Detecting Endotracheal Tube and Carina on Portable Supine Chest Radiographs using One-Stage Detector with a Coarse-To-Fine Attention

<div>
<div align="center">
    <a href='' target='_blank'>Liang-Kai Mao<sup>1</sup></a>&emsp;
    <a href='' target='_blank'>Min-Hsin Huang<sup>2</sup></a>&emsp;
    <a href='' target='_blank'>Chao-Han Lai<sup>2</sup></a>&emsp;
    </br>
    <a href='' target='_blank'>Yung-Nien Sun<sup>1</sup></a>&emsp;
    <a href='' target='_blank'>Chi-Yeh Chen <sup>1,*</sup></a>&emsp;
</div>
<div>
<br>
<div>
    <sup>1</sup>Department of Computer Science and Information Engineering, National Cheng Kung University, Taiwan</a>&emsp;
    </br>
    <sup>2</sup>Department of Surgery, National Cheng Kung University Hospital, College of Medicine, National Cheng Kung University, Taiwan
    </a></br>
    <sup>*</sup>Author to whom correspondence should be addressed.&emsp;
</div>
<br>

[![MDPI](https://img.shields.io/badge/MDPI-2075.4418-b31b1b?style=plastic&color=b31b1b&link=https://www.mdpi.com/2075-4418/12/8/1913)](https://www.mdpi.com/2075-4418/12/8/1913)
---

## **Folder structure**
```
├── README.md                | 
├── environment.yaml         | environment
├── classes.txt              | 
├── coco_eval.py             | evaluate detection result in coco format
├── detect_ccy.py            | detect malposition, saving detection result and summary   
├── resnet50.pth             | downloaded from pytorch
├── run_all.py               | integrate the commands    
├── toanno.py                | generate annotations
├── train_coco.py            | training 
├── checkpoint               | save weight
------------------------------ 
├── $coco_test$              | your own dataset
│   ├── annotations          | your own annotation file
│   ├── images_dcm           | your own dicom CXRs 
│       ├── train            | 
│       ├── test             | 
│       └── val              | 
│   └── labels               | your own labels
│       ├── train            | 
│       ├── test             | 
│       └── val              |
------------------------------
├── dataset                  | 
│   ├── .DS_Store            | 
│   ├── __init__.py          | 
│   ├── augment.py           | 
│   └── COCO_dataset.py      | 
------------------------------
├── model                    | FCOS
│   ├── .DS_Store            | 
│   ├── __init__.py          | 
│   ├── config.py            | 
│   ├── fcos.py              | 
│   ├── fpn_neck.py          | neak
│   ├── head.py              | head
│   ├── loss.py              | loss function
│   ├── util.py              | some util processes and attention modules
│   └── backbone             | backbone
│       ├── __init__.py      | 
│       └── resnet.py        |
------------------------------
├── runs                     | automatically generate, tensorboard record
├── out_images               | automatically generate, detection result  
├── coco_bbox_results        | automatically generate, bbox detection result 
└── losses_ccy.xlsx          | automatically generate, summary 
```
---
<br>

## **Requirements**
```
conda env create -f environment.yaml -n $your virtual environment$
```
Change the text in the $$ to your own setting.
#### **Or install**
* pytorch == 1.9
* CUDA == 10.2
* opencv 
* pydicom
* tensorboardX
* scipy
* matplotlib
* pycocotools
* tqdm
* openpyxl
* skimage
---
#### <span class="red">Note</span>

if your CUDA version is 11.x and face the RuntimeError bellow:
```
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
```
The reason may be out of cuda memory, please down the batch size or image resolution.

<br>

## **The entire process (run_all.py)**
1. run ''toanno.py'' to generate annotations.
2. run ''train_coco.py'' to train the model.
3. run ''coco_eval.py'' if you want to know the results of bbox.
4. run ''detect_ccy.py'' to generate the detection results.
---
<br>

## **Example**
#### **Make annotation** 
```
python toanno.py -l=./coco_test/labels -c=./classes.txt -a=./coco_test/annotations -m=train
```
if you have any problems please type:
```
python toanno.py -h
```
---
#### **Evaluate example**
Change the text in the $$ to your own setting.
```
python coco_eval.py -t=./$coco_test$/images_dcm/test -a=./$coco_test$/annotations/instances_test_dcm.json -w=./checkpoint/final_120.pth
```
if you have any problems please type:
```
python coco_eval.py -h
```
---
#### **Detection example**
Change the text in the $$ to your own setting.
```
python detect_ccy.py -t=./$coco_test$/images_dcm/test -a=./$coco_test$/annotations/instances_test_dcm.json -w=./checkpoint/final_120.pth
```
if you have any problems please type:
```
python detect_ccy.py -h
```
---
#### **Training example**
Change the text in the $$ to your own setting.
```
python train_coco.py -e=120 -b=2 -d=./$coco_test$/images_dcm/train -a=./$coco_test$/annotations/instances_train_dcm.json
```
if you have any problems please type:
```
python train_coco.py -h
```

## **Citation**
If the code and paper help your research, please kindly cite:
```
@article{mao2022detecting,
  title={Detecting endotracheal tube and carina on portable supine chest radiographs using one-stage detector with a coarse-to-fine attention},
  author={Mao, Liang-Kai and Huang, Min-Hsin and Lai, Chao-Han and Sun, Yung-Nien and Chen, Chi-Yeh},
  journal={Diagnostics},
  volume={12},
  number={8},
  pages={1913},
  year={2022},
  publisher={MDPI}
}
```

## **Acknowledgement**
The research was supported by the Higher Education Sprout Project, Ministry of Education to the Headquarters of University Advancement at National Cheng Kung University (NCKU) and also by the Ministry of Science and Technology, Executive Yuan, Taiwan (MOST 111-2221-E-006-125-MY2 and MOST 109-2634-F-006-023), by National Cheng Kung University Hospital, Tainan, Taiwan (NCKUH-10901003).