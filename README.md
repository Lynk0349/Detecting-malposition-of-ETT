# Detecting Endotracheal Tube and Carina on Portable Supine Chest Radiographs using One-Stage Detector with a Coarse-To-Fine Attention

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


